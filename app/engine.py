"""Candidate generation + ranking engine (ATP).

Current implementation goal:
- Pull Odds API events.
- Normalize ML (h2h) prices.
- Build an Elo-first probability baseline.
- Score every candidate on three separate axes:
  - Matchup Strength
  - Market Value
  - Reliability
- Persist explainable component scores + reasons for UI/debugging.

This keeps the product tennis-first:
- player-vs-player strengths and weaknesses drive the matchup view
- market comparison decides whether the price is actionable
- reliability decides how much we should trust the read
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json
import math
import sqlite3

from .modeling import implied_prob_from_decimal, ev_per_1u_risk, elo_to_p
from .elo import ensure_atp_surface_elo, get_player_surface_elo


# ---------- Style / feature lookups ----------

def get_mcp_style_profile(conn, player_name: str):
    """Return MCP aggregated style profile for a player name (men)."""
    if not player_name:
        return None
    row = conn.execute(
        "SELECT * FROM style_mcp_player_m WHERE player = ?",
        (player_name,),
    ).fetchone()
    return row


def get_recent_profile(conn, player_id: int | None):
    if player_id is None:
        return None
    try:
        return conn.execute(
            "SELECT * FROM atp_recent_oppq_10 WHERE player_id = ?",
            (player_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None


def get_surface_split(conn, player_id: int | None, surface: str | None):
    if player_id is None or not surface:
        return None
    try:
        return conn.execute(
            "SELECT * FROM atp_surface_splits WHERE player_id = ? AND surface = ?",
            (player_id, surface),
        ).fetchone()
    except sqlite3.OperationalError:
        return None


def get_head_to_head(conn, player_a_id: int | None, player_b_id: int | None):
    if player_a_id is None or player_b_id is None:
        return None
    try:
        row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN winner_id = ? AND loser_id = ? THEN 1 ELSE 0 END) AS a_wins,
              SUM(CASE WHEN winner_id = ? AND loser_id = ? THEN 1 ELSE 0 END) AS b_wins,
              COUNT(*) AS matches
            FROM ta_matches
            WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
            """,
            (player_a_id, player_b_id, player_b_id, player_a_id, player_a_id, player_b_id, player_b_id, player_a_id),
        ).fetchone()
        return row
    except sqlite3.OperationalError:
        return None


# ---------- Small math helpers ----------

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _to_0_100_from_centered(delta: float, scale: float) -> float:
    if scale <= 0:
        return 50.0
    return _clamp(50.0 + 50.0 * math.tanh(delta / scale))


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _best_h2h_price(event: dict[str, Any], side: str) -> tuple[float | None, str | None]:
    """Return best decimal price and book title for a given team/player name."""
    best_p = None
    best_book = None
    for bm in (event.get("bookmakers") or []):
        title = bm.get("title")
        for m in (bm.get("markets") or []):
            if m.get("key") != "h2h":
                continue
            for o in (m.get("outcomes") or []):
                if o.get("name") != side:
                    continue
                p = o.get("price")
                if not isinstance(p, (int, float)):
                    continue
                if best_p is None or p > best_p:
                    best_p = float(p)
                    best_book = title
    return best_p, best_book


def _best_h2h_price_from_payload(payload_json: str | None, side: str) -> float | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
    except Exception:
        return None
    price, _book = _best_h2h_price(payload, side)
    return price


# ---------- Explainable matchup logic ----------

def mcp_style_edge(conn, player: str, opp: str) -> tuple[float | None, dict[str, float], list[str]]:
    """Compute a small, explainable style edge from MCP aggregated profiles.

    Returns:
      edge: signed scalar (player - opp), None if insufficient data
      components: raw component deltas used for the edge
      reasons: short explanation strings
    """
    p = get_mcp_style_profile(conn, player)
    o = get_mcp_style_profile(conn, opp)
    if not p or not o:
        return None, {}, ["MCP style: missing profile(s)."]

    p_matches = int(p["matches"] or 0)
    o_matches = int(o["matches"] or 0)
    if min(p_matches, o_matches) < 5:
        return None, {}, [f"MCP style: insufficient coverage (matches {p_matches} vs {o_matches})."]

    # first serve win%, winner rate, ace rate, return win% are the current core inputs.
    w = {
        "first_win_pct": 1.0,
        "return_win_pct": 1.0,
        "ace_rate": 0.7,
        "winner_rate": 0.6,
        "df_rate": -0.7,
        "unforced_rate": -0.5,
    }

    def val(row, k):
        x = row[k]
        return float(x) if x is not None else 0.0

    edge = 0.0
    components = {}
    for k, wk in w.items():
        dv = val(p, k) - val(o, k)
        components[k] = dv
        edge += wk * dv

    reasons = [
        f"MCP style coverage: {p_matches} vs {o_matches} matches.",
        "MCP style deltas (player-opp): "
        + ", ".join([
            f"{k} {components[k]:+.3f}" for k in ["first_win_pct", "return_win_pct", "ace_rate", "winner_rate", "df_rate", "unforced_rate"]
        ]),
        f"MCP style edge (weighted): {edge:+.3f}",
    ]

    return edge, components, reasons


def _market_movement_score(conn, match_id: str, side: str, current_price: float | None) -> tuple[float, str, dict[str, float | None]]:
    """Score open->current movement using earliest stored snapshot for the match.

    We use best available price from the earliest snapshot as the 'open' proxy.
    If unavailable, return a neutral score and mark it as missing.
    """
    if current_price is None:
        return 50.0, "No current price available for market movement.", {"open_price": None, "current_price": None, "price_delta": None}

    row = conn.execute(
        "SELECT payload_json FROM odds_snapshots WHERE match_id = ? ORDER BY ts ASC, id ASC LIMIT 1",
        (match_id,),
    ).fetchone()
    open_price = _best_h2h_price_from_payload(row["payload_json"], side) if row else None
    if open_price is None:
        return 50.0, "Open/current movement unavailable yet — using neutral market movement score.", {"open_price": None, "current_price": current_price, "price_delta": None}

    # Better current price than open = positive movement for our side.
    delta = current_price - open_price
    score = _to_0_100_from_centered(delta, 0.20)
    reason = f"Open/current price move: {open_price:.2f} → {current_price:.2f} ({delta:+.2f} dec)."
    return score, reason, {"open_price": open_price, "current_price": current_price, "price_delta": delta}


@dataclass
class Candidate:
    match_id: str
    commence_time: str | None
    tournament: str | None
    surface: str | None
    player_a: str
    player_b: str
    market_type: str  # ML|SPREAD|TOTAL
    side: str
    line: float | None
    price_decimal: float | None
    book: str | None

    p0: float | None
    p_final: float | None
    q_implied: float | None
    ev: float | None
    confidence: float | None
    ev_adj: float | None

    matchup_strength: float | None = None
    market_value: float | None = None
    reliability: float | None = None
    component_scores: dict[str, float | None] = field(default_factory=dict)
    axis_notes: dict[str, list[str]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# ---------- Main candidate generation ----------

def generate_ml_candidates(conn, events: list[dict[str, Any]], surface: str | None, player_id_lookup, top_n: int | None = None) -> list[Candidate]:
    """Generate ML candidates with three-axis scoring.

    Matchup Strength
    - player quality
    - surface strength
    - recent form
    - serve/return profile
    - style interaction

    Market Value
    - open/current comparison
    - implied vs fair probability
    - price-bucket viability

    Reliability
    - sample size
    - calibration proxy (model/market agreement for now)
    - injury/uncertainty proxy (neutral until data exists)
    - data completeness
    """
    try:
        ensure_atp_surface_elo(conn)
    except sqlite3.OperationalError:
        # Fresh deploy before historical bootstrap: run market-first fallback instead of crashing.
        pass

    out: list[Candidate] = []
    for e in events:
        away = e.get("away_team")
        home = e.get("home_team")
        if not away or not home:
            continue
        match_id = e.get("id") or f"{e.get('commence_time','')}-{away}-{home}"
        start = e.get("commence_time")

        away_price, away_book = _best_h2h_price(e, away)
        home_price, home_book = _best_h2h_price(e, home)
        if away_price is None or home_price is None:
            continue

        # Market implied from best prices.
        q_away = implied_prob_from_decimal(away_price) or 0.5
        q_home = implied_prob_from_decimal(home_price) or 0.5
        total = q_away + q_home
        p_mkt_away = q_away / total if total > 0 else 0.5
        p_mkt_home = q_home / total if total > 0 else 0.5

        away_id = player_id_lookup(conn, away)
        home_id = player_id_lookup(conn, home)

        elo_away, n_away = get_player_surface_elo(conn, away_id, surface)
        elo_home, n_home = get_player_surface_elo(conn, home_id, surface)
        away_recent = get_recent_profile(conn, away_id)
        home_recent = get_recent_profile(conn, home_id)
        away_split = get_surface_split(conn, away_id, surface)
        home_split = get_surface_split(conn, home_id, surface)
        h2h = get_head_to_head(conn, away_id, home_id)

        use_elo = (elo_away is not None) and (elo_home is not None)
        if use_elo:
            p0_away = elo_to_p(float(elo_away) - float(elo_home))
            p0_home = 1.0 - p0_away
            reason0 = f"Surface Elo baseline ({surface or 'n/a'}): Elo {float(elo_away):.0f} vs {float(elo_home):.0f}."
        else:
            p0_away = p_mkt_away
            p0_home = p_mkt_home
            reason0 = "Fallback baseline: normalized market implied probabilities (player mapping / Elo missing)."

        # Style edges are side-specific.
        away_style_edge, away_style_components, away_style_reasons = mcp_style_edge(conn, away, home)
        home_style_edge, home_style_components, home_style_reasons = mcp_style_edge(conn, home, away)

        # Shared matchup ingredients.
        elo_diff = (float(elo_away) - float(elo_home)) if use_elo else 0.0
        away_surface_win = _safe_float(away_split["win_pct"], None) if away_split else None
        home_surface_win = _safe_float(home_split["win_pct"], None) if home_split else None
        away_recent_win = _safe_float(away_recent["win_pct_n"], None) if away_recent else None
        home_recent_win = _safe_float(home_recent["win_pct_n"], None) if home_recent else None
        away_opp_rank = _safe_float(away_recent["avg_opp_rank"], None) if away_recent else None
        home_opp_rank = _safe_float(home_recent["avg_opp_rank"], None) if home_recent else None
        away_style = get_mcp_style_profile(conn, away)
        home_style = get_mcp_style_profile(conn, home)

        # Head-to-head only as a small context note.
        h2h_reason = None
        if h2h and int(h2h["matches"] or 0) > 0:
            h2h_reason = f"Head-to-head: {away} {int(h2h['a_wins'] or 0)} - {int(h2h['b_wins'] or 0)} {home} across {int(h2h['matches'] or 0)} matches."

        def recent_quality_score(player_recent, opp_recent):
            if not player_recent or not opp_recent:
                return 50.0, "Recent form: insufficient recent-opponent-quality data."
            win_delta = (_safe_float(player_recent["win_pct_n"], 0.5) - _safe_float(opp_recent["win_pct_n"], 0.5))
            rank_delta = (_safe_float(opp_recent["avg_opp_rank"], 100.0) - _safe_float(player_recent["avg_opp_rank"], 100.0))
            score = 50.0 + 28.0 * math.tanh(win_delta / 0.10) + 16.0 * math.tanh(rank_delta / 35.0)
            score = _clamp(score)
            reason = (
                f"Recent form (last 10): win% {_safe_float(player_recent['win_pct_n'],0)*100:.1f}% vs {_safe_float(opp_recent['win_pct_n'],0)*100:.1f}% "
                f"| avg opp rank {_safe_float(player_recent['avg_opp_rank'],0):.1f} vs {_safe_float(opp_recent['avg_opp_rank'],0):.1f}."
            )
            return score, reason

        def serve_return_score(player_style, opp_style, player_name, opp_name):
            if not player_style or not opp_style:
                return 50.0, f"Serve/return profile: insufficient MCP data for {player_name} vs {opp_name}."
            p_first = _safe_float(player_style["first_win_pct"], 0.0)
            o_first = _safe_float(opp_style["first_win_pct"], 0.0)
            p_ret = _safe_float(player_style["return_win_pct"], 0.0)
            o_ret = _safe_float(opp_style["return_win_pct"], 0.0)
            p_ace = _safe_float(player_style["ace_rate"], 0.0)
            o_ace = _safe_float(opp_style["ace_rate"], 0.0)
            p_df = _safe_float(player_style["df_rate"], 0.0)
            o_df = _safe_float(opp_style["df_rate"], 0.0)
            score = 50.0
            score += 18.0 * math.tanh((p_first - o_first) / 0.06)
            score += 18.0 * math.tanh((p_ret - o_ret) / 0.05)
            score += 10.0 * math.tanh((p_ace - o_ace) / 0.03)
            score -= 10.0 * math.tanh((p_df - o_df) / 0.02)
            score = _clamp(score)
            reason = (
                f"Serve/return profile: 1st-win {p_first*100:.1f}% vs {o_first*100:.1f}%, "
                f"return-win {p_ret*100:.1f}% vs {o_ret*100:.1f}%, ace-rate {p_ace:.3f} vs {o_ace:.3f}."
            )
            return score, reason

        def side_scores(side_name, opp_name, price, book, p0, p_mkt, style_edge, style_components, style_reasons, player_id, opp_id, elo_self, elo_opp, n_self, n_opp, recent_self, recent_opp, split_self, split_opp, style_self, style_opp):
            q = implied_prob_from_decimal(price)
            ev = ev_per_1u_risk(p0, price)
            edge = (p0 or 0.5) - (p_mkt or 0.5)

            # ----- Matchup Strength -----
            player_quality = _to_0_100_from_centered((float(elo_self or 1500) - float(elo_opp or 1500)), 140.0)
            surface_strength = 50.0
            if split_self and split_opp:
                surface_strength = 50.0 + 30.0 * math.tanh(((_safe_float(split_self["win_pct"], 0.5) - _safe_float(split_opp["win_pct"], 0.5))) / 0.10)
                surface_strength = _clamp(surface_strength)
                surface_reason = (
                    f"Surface strength ({surface or 'n/a'}): win% {_safe_float(split_self['win_pct'],0)*100:.1f}% vs {_safe_float(split_opp['win_pct'],0)*100:.1f}% "
                    f"| matches {int(split_self['matches'] or 0)} vs {int(split_opp['matches'] or 0)}."
                )
            else:
                surface_reason = f"Surface strength ({surface or 'n/a'}): insufficient split data."

            recent_form, recent_reason = recent_quality_score(recent_self, recent_opp)
            serve_return, sr_reason = serve_return_score(style_self, style_opp, side_name, opp_name)
            style_interaction = 50.0
            style_reason = "Style interaction: insufficient matchup-specific style coverage."
            if style_edge is not None:
                style_interaction = _to_0_100_from_centered(style_edge, 0.03)
                style_reason = style_reasons[-1]

            matchup_strength = _clamp(
                0.26 * player_quality
                + 0.24 * surface_strength
                + 0.20 * recent_form
                + 0.18 * serve_return
                + 0.12 * style_interaction
            )

            # ----- Market Value -----
            movement_score, movement_reason, movement_meta = _market_movement_score(conn, match_id, side_name, price)
            fair_edge_score = _to_0_100_from_centered(edge, 0.06)
            # Prefer coinflip to modest favorite/dog range; longshots / heavy favs get a haircut.
            if price is None:
                price_bucket = 50.0
                price_reason = "Price bucket viability: price unavailable."
            elif price <= 1.20:
                price_bucket = 30.0
                price_reason = f"Price bucket viability: very heavy favorite at {price:.2f} — needs stronger edge expression."
            elif price <= 1.55:
                price_bucket = 58.0
                price_reason = f"Price bucket viability: favorite range at {price:.2f}."
            elif price <= 2.40:
                price_bucket = 76.0
                price_reason = f"Price bucket viability: strong working range at {price:.2f}."
            elif price <= 3.50:
                price_bucket = 60.0
                price_reason = f"Price bucket viability: dog range at {price:.2f} — edge can work but gets noisier."
            else:
                price_bucket = 38.0
                price_reason = f"Price bucket viability: longshot range at {price:.2f} — historically more fragile."

            market_value = _clamp(0.30 * movement_score + 0.50 * fair_edge_score + 0.20 * price_bucket)

            # ----- Reliability -----
            sample_parts = []
            if n_self is not None:
                sample_parts.append(min(float(n_self), 80.0) / 80.0)
            if n_opp is not None:
                sample_parts.append(min(float(n_opp), 80.0) / 80.0)
            if style_self and style_opp:
                sample_parts.append(min(float(style_self["matches"] or 0), float(style_opp["matches"] or 0), 50.0) / 50.0)
            if recent_self and recent_opp:
                sample_parts.append(min(float(recent_self["n"] or 0), float(recent_opp["n"] or 0), 10.0) / 10.0)
            sample_size = _clamp((sum(sample_parts) / len(sample_parts)) * 100.0) if sample_parts else 25.0
            sample_reason = f"Sample size support: {sample_size:.0f}/100 from Elo/style/recent coverage."

            # Until we ship formal calibration tables, use model-market agreement as a calibration proxy.
            d = abs((p0 or 0.5) - (p_mkt or 0.5))
            calibration = _clamp(100.0 * math.exp(-5.0 * d))
            calibration_reason = f"Calibration proxy: model/market delta {d:.3f} -> agreement score {calibration:.0f}/100."

            injury_uncertainty = 55.0
            injury_reason = "Injury/uncertainty: no structured injury feed yet, so this is neutral-to-cautious by default."

            available_components = [
                use_elo,
                split_self is not None and split_opp is not None,
                recent_self is not None and recent_opp is not None,
                style_self is not None and style_opp is not None,
            ]
            data_completeness = _clamp(100.0 * (sum(1 for x in available_components if x) / len(available_components)))
            data_reason = f"Data completeness: {sum(1 for x in available_components if x)}/{len(available_components)} core components available."

            reliability = _clamp(0.35 * sample_size + 0.30 * calibration + 0.10 * injury_uncertainty + 0.25 * data_completeness)

            # Confidence now comes from matchup strength + reliability, with a light market sanity check.
            confidence = _clamp(0.45 * matchup_strength + 0.45 * reliability + 0.10 * market_value) / 100.0
            ev_adj = (ev * confidence) if ev is not None else None

            components = {
                "player_quality": round(player_quality, 2),
                "surface_strength": round(surface_strength, 2),
                "recent_form": round(recent_form, 2),
                "serve_return_profile": round(serve_return, 2),
                "style_interaction": round(style_interaction, 2),
                "open_close_comparison": round(movement_score, 2),
                "implied_vs_fair_probability": round(fair_edge_score, 2),
                "price_bucket_viability": round(price_bucket, 2),
                "sample_size": round(sample_size, 2),
                "calibration": round(calibration, 2),
                "injury_uncertainty": round(injury_uncertainty, 2),
                "data_completeness": round(data_completeness, 2),
                "model_edge_prob": round(edge, 4),
                "open_price_decimal": movement_meta.get("open_price"),
                "current_price_decimal": movement_meta.get("current_price"),
                "price_delta_decimal": movement_meta.get("price_delta"),
            }
            if style_components:
                components["style_component_deltas"] = {k: round(v, 4) for k, v in style_components.items()}

            axis_notes = {
                "matchup_strength": [
                    reason0,
                    surface_reason,
                    recent_reason,
                    sr_reason,
                    style_reason,
                ],
                "market_value": [
                    movement_reason,
                    f"Implied vs fair probability: model {(p0 or 0.5)*100:.1f}% vs market {(p_mkt or 0.5)*100:.1f}% -> edge {edge*100:+.1f} pts.",
                    price_reason,
                ],
                "reliability": [
                    sample_reason,
                    calibration_reason,
                    injury_reason,
                    data_reason,
                ],
            }

            reasons = [
                f"Matchup Strength {matchup_strength:.0f}/100 • Market Value {market_value:.0f}/100 • Reliability {reliability:.0f}/100.",
                f"Player quality edge: {player_quality:.0f}/100 from Elo baseline vs opponent.",
                surface_reason,
                recent_reason,
                sr_reason,
                style_reason,
                movement_reason,
                price_reason,
                calibration_reason,
            ]
            if h2h_reason:
                reasons.append(h2h_reason)

            return Candidate(
                match_id=match_id,
                commence_time=start,
                tournament=None,
                surface=surface,
                player_a=away,
                player_b=home,
                market_type="ML",
                side=side_name,
                line=None,
                price_decimal=price,
                book=book,
                p0=p0,
                p_final=p0,
                q_implied=q,
                ev=ev,
                confidence=confidence,
                ev_adj=ev_adj,
                matchup_strength=matchup_strength,
                market_value=market_value,
                reliability=reliability,
                component_scores=components,
                axis_notes=axis_notes,
                reasons=reasons[:10],
            )

        out.append(
            side_scores(
                away, home, away_price, away_book, p0_away, p_mkt_away,
                away_style_edge, away_style_components, away_style_reasons,
                away_id, home_id, elo_away, elo_home, n_away, n_home,
                away_recent, home_recent, away_split, home_split, away_style, home_style,
            )
        )
        out.append(
            side_scores(
                home, away, home_price, home_book, p0_home, p_mkt_home,
                home_style_edge, home_style_components, home_style_reasons,
                home_id, away_id, elo_home, elo_away, n_home, n_away,
                home_recent, away_recent, home_split, away_split, home_style, away_style,
            )
        )

    out.sort(key=lambda c: (c.ev_adj if c.ev_adj is not None else -999), reverse=True)
    if top_n is None or top_n <= 0:
        return out
    return out[:top_n]
