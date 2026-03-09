"""Candidate generation + ranking engine (ATP).

v0 implementation goal:
- Pull Odds API events.
- Normalize ML (h2h) prices.
- Compute baseline p0 from a placeholder rating (until Elo table exists).
- Compute EV vs best price, attach confidence stub.
- Persist Top N candidates for both Actionable and Debug views.

We will progressively replace placeholders with:
- surface Match Elo baseline
- serve/return surface ratings + shrinkage
- recency OQW + hotness
- overlays with cap
- spreads/totals + heavy-favorite substitution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

from .modeling import implied_prob_from_decimal, ev_per_1u_risk, elo_to_p
from .elo import ensure_atp_surface_elo, get_player_surface_elo


def get_mcp_style_profile(conn, player_name: str):
    """Return MCP aggregated style profile for a player name (men).

    MCP coverage is partial; we gate by matches in the caller.
    """
    if not player_name:
        return None
    row = conn.execute(
        "SELECT * FROM style_mcp_player_m WHERE player = ?",
        (player_name,),
    ).fetchone()
    return row


def mcp_style_edge(conn, player: str, opp: str) -> tuple[float | None, list[str]]:
    """Compute a small, explainable style edge from MCP aggregated profiles.

    Returns:
      edge: signed scalar (player - opp), None if insufficient data
      reasons: short explanation strings

    NOTE: This is used for *confidence/ranking* first, not probability p_final.
    """
    p = get_mcp_style_profile(conn, player)
    o = get_mcp_style_profile(conn, opp)
    if not p or not o:
        return None, ["MCP style: missing profile(s)."]

    p_matches = int(p["matches"] or 0)
    o_matches = int(o["matches"] or 0)
    if min(p_matches, o_matches) < 5:
        return None, [f"MCP style: insufficient coverage (matches {p_matches} vs {o_matches})."]

    # Weights chosen from our quick MCP scan (winner vs loser separation):
    # first serve win%, winner rate, ace rate, return win% are most informative.
    w = {
        "first_win_pct": 1.0,
        "return_win_pct": 1.0,
        "ace_rate": 0.7,
        "winner_rate": 0.6,
        "df_rate": -0.7,
        "unforced_rate": -0.5,
        # net_freq/net_win_pct can matter, but are more matchup-dependent; defer for now.
    }

    def val(row, k):
        x = row[k]
        return float(x) if x is not None else 0.0

    # raw edge in "rate units"
    edge = 0.0
    contrib = {}
    for k, wk in w.items():
        dv = val(p, k) - val(o, k)
        contrib[k] = wk * dv
        edge += wk * dv

    reasons = [
        f"MCP style coverage: {p_matches} vs {o_matches} matches.",
        "MCP style deltas (player-opp): "
        + ", ".join([
            f"{k} {val(p,k)-val(o,k):+.3f}" for k in ["first_win_pct","return_win_pct","ace_rate","winner_rate","df_rate","unforced_rate"]
        ]),
        f"MCP style edge (weighted): {edge:+.3f}",
    ]

    return edge, reasons


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

    reasons: list[str]


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


def generate_ml_candidates(conn, events: list[dict[str, Any]], surface: str | None, player_id_lookup, top_n: int = 10) -> list[Candidate]:
    """Generate ML candidates only.

    v1 baseline:
    - p0 from ATP surface Elo (built from Tennis Abstract match history).
    - fallback to market-implied normalized baseline if we can't map players.

    Confidence v1:
    - increases with surface Elo match sample.
    - decreases when model is far from market implied (anti-hallucination guard).
    """
    ensure_atp_surface_elo(conn)

    out: list[Candidate] = []
    for e in events:
        away = e.get("away_team")
        home = e.get("home_team")
        if not away or not home:
            continue
        match_id = e.get("id") or f"{e.get('commence_time','')}-{away}-{home}"
        start = e.get("commence_time")

        # best prices
        away_price, away_book = _best_h2h_price(e, away)
        home_price, home_book = _best_h2h_price(e, home)
        if away_price is None or home_price is None:
            continue

        # Market implied (best prices)
        q_away = implied_prob_from_decimal(away_price) or 0.5
        q_home = implied_prob_from_decimal(home_price) or 0.5
        total = q_away + q_home
        p_mkt_away = q_away / total if total > 0 else 0.5
        p_mkt_home = q_home / total if total > 0 else 0.5

        # Elo baseline if we can map player ids
        away_id = player_id_lookup(conn, away)
        home_id = player_id_lookup(conn, home)
        elo_away, n_away = get_player_surface_elo(conn, away_id, surface)
        elo_home, n_home = get_player_surface_elo(conn, home_id, surface)

        use_elo = (elo_away is not None) and (elo_home is not None)
        if use_elo:
            p0_away = elo_to_p(float(elo_away) - float(elo_home))
            p0_home = 1.0 - p0_away
            reason0 = f"Surface Elo baseline ({surface}): Elo {elo_away:.0f} vs {elo_home:.0f}."
        else:
            p0_away = p_mkt_away
            p0_home = p_mkt_home
            reason0 = "Fallback baseline: normalized market implied probs (could not map players to TA IDs)."

        # Confidence v1
        # sample component from surface Elo match counts (cap at 60 matches)
        n_min = min(n_away or 0, n_home or 0)
        sample_conf = min(1.0, n_min / 60.0) if use_elo else 0.25

        # agreement component vs market (penalize large deltas)
        # delta 0.10 -> ~0.61, delta 0.20 -> ~0.37
        def agree(p_model, p_mkt):
            d = abs((p_model or 0.5) - (p_mkt or 0.5))
            return math.exp(-5.0 * d)

        agree_away = agree(p0_away, p_mkt_away)
        agree_home = agree(p0_home, p_mkt_home)

        # final confidence base
        base_conf = 0.20 + 0.60 * sample_conf

        for side, price, book, p0, p_mkt, agree_c, opp_name in [
            (away, away_price, away_book, p0_away, p_mkt_away, agree_away, home),
            (home, home_price, home_book, p0_home, p_mkt_home, agree_home, away),
        ]:
            q = implied_prob_from_decimal(price)
            ev = ev_per_1u_risk(p0, price)

            conf = max(0.0, min(1.0, base_conf * (0.50 + 0.50 * agree_c)))

            # Style overlay (A-phase): affects confidence/ranking + explanations only.
            # Small, capped bump so we don't override Elo/market sanity checks.
            style_edge, style_reasons = mcp_style_edge(conn, side, opp_name)
            if style_edge is not None:
                # Scale heuristic: typical edge magnitudes are small (~0.00-0.05).
                bump = 0.10 * math.tanh(style_edge / 0.03)
                conf = max(0.0, min(1.0, conf * (1.0 + bump)))

            ev_adj = (ev * conf) if ev is not None else None

            reasons = [reason0]
            if use_elo:
                reasons.append(f"Surface Elo sample (min matches): {n_min} → sample_conf {sample_conf:.2f}.")
            reasons.append(f"Model vs market delta: {abs(p0 - p_mkt):.3f} → agree {agree_c:.2f}.")
            if style_edge is not None:
                reasons.append(style_reasons[-1])

            out.append(
                Candidate(
                    match_id=match_id,
                    commence_time=start,
                    tournament=None,
                    surface=surface,
                    player_a=away,
                    player_b=home,
                    market_type="ML",
                    side=side,
                    line=None,
                    price_decimal=price,
                    book=book,
                    p0=p0,
                    p_final=p0,
                    q_implied=q,
                    ev=ev,
                    confidence=conf,
                    ev_adj=ev_adj,
                    reasons=reasons[:4],
                )
            )

    out.sort(key=lambda c: (c.ev_adj if c.ev_adj is not None else -999), reverse=True)
    return out[:top_n]
