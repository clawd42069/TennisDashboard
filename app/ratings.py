"""Lean/Tier computation (v1).

Weights: 60% opponent-quality progression, 40% surface.

This is intentionally simple to start; we’ll iterate once we see outputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rating:
    lean: str
    confidence: float  # 0-100
    tier: str          # S/A/B/C
    reasons: list[str]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def tier_from_conf(conf: float) -> str:
    if conf >= 80:
        return "S"
    if conf >= 70:
        return "A"
    if conf >= 60:
        return "B"
    return "C"


def score_progression(player_avg_opp_rank: float | None, opp_avg_opp_rank: float | None) -> tuple[float, list[str]]:
    """Lower avg opponent rank = harder schedule (better signal)."""
    if player_avg_opp_rank is None or opp_avg_opp_rank is None:
        return 50.0, ["Progression: insufficient opponent-rank data (default 50)."]

    # Map rank to 0-100 score: 1 is elite, 200 is weak
    def rank_to_score(r):
        r = clamp(r, 1, 200)
        return 100 - ((r - 1) / 199) * 100

    s1 = rank_to_score(player_avg_opp_rank)
    s2 = rank_to_score(opp_avg_opp_rank)
    delta = s1 - s2

    reasons = [
        f"Opp quality (last 10): avg opp rank {player_avg_opp_rank:.1f} vs {opp_avg_opp_rank:.1f} (harder schedule = bullish)."
    ]

    # Convert delta to advantage: +/- 20 points max
    adv = clamp(50 + delta * 0.4, 0, 100)
    return adv, reasons


def score_surface(player_win_pct: float | None, opp_win_pct: float | None, surface: str | None) -> tuple[float, list[str]]:
    if player_win_pct is None or opp_win_pct is None:
        return 50.0, [f"Surface ({surface or 'n/a'}): insufficient surface split data (default 50)."]

    delta = (player_win_pct - opp_win_pct) * 100
    adv = clamp(50 + delta * 0.8, 0, 100)
    return adv, [f"Surface ({surface or 'n/a'}): win% {player_win_pct*100:.1f}% vs {opp_win_pct*100:.1f}%."
    ]


def rate_match(
    away: str,
    home: str,
    surface: str | None,
    away_prog: float | None,
    home_prog: float | None,
    away_surf_win: float | None,
    home_surf_win: float | None,
    w_progression: float = 0.60,
    w_surface: float = 0.40,
) -> Rating:
    # progression score per side is based on avg opp rank; we compute advantage score for away vs home
    prog_score, prog_reasons = score_progression(away_prog, home_prog)
    surf_score, surf_reasons = score_surface(away_surf_win, home_surf_win, surface)

    conf = clamp((prog_score * w_progression) + (surf_score * w_surface), 0, 100)

    lean = away if conf >= 50 else home
    # Recenter confidence around 50: distance from 50 indicates strength
    strength = abs(conf - 50) * 2  # 0..100
    tier = tier_from_conf(50 + strength / 2)

    reasons = []
    reasons.extend(prog_reasons)
    reasons.extend(surf_reasons)
    reasons.append(f"Weights: {int(w_progression*100)}/{int(w_surface*100)} (progression/surface).")

    return Rating(lean=lean, confidence=float(50 + strength / 2), tier=tier, reasons=reasons[:3])
