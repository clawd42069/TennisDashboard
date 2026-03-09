"""Modeling utilities for Tennis Analyst (ATP).

This module implements the Elo-first baseline probability + conversions.
Overlays (serve/return, recency, matchup flags) are layered later.

Design targets (from spec):
- Baseline p0 from surface-adjusted match Elo.
- Bo3/Bo5 conversion via per-set probability.
- EV computed vs best available price.
"""

from __future__ import annotations

import math


def elo_to_p(elo_diff: float, scale: float = 400.0) -> float:
    """Convert Elo difference to win probability."""
    return 1.0 / (1.0 + 10 ** (-elo_diff / scale))


def logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def bo3_match_p_from_set_p(s: float) -> float:
    # P(win best-of-3) = s^2(3-2s)
    return (s * s) * (3.0 - 2.0 * s)


def bo5_match_p_from_set_p(s: float) -> float:
    # P(win best-of-5) = s^3(10 - 15s + 6s^2)
    return (s ** 3) * (10.0 - 15.0 * s + 6.0 * (s ** 2))


def solve_set_p_from_bo3_match_p(p_bo3: float, iters: int = 40) -> float:
    """Numerically solve for set-win probability s given bo3 match probability.

    Monotonic on [0,1], so binary search is fine.
    """
    p_bo3 = min(1 - 1e-9, max(1e-9, p_bo3))
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        pm = bo3_match_p_from_set_p(mid)
        if pm < p_bo3:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def convert_bo3_to_bo5(p_bo3: float) -> float:
    s = solve_set_p_from_bo3_match_p(p_bo3)
    return bo5_match_p_from_set_p(s)


def implied_prob_from_decimal(d: float | None) -> float | None:
    if not d or d <= 1.0:
        return None
    return 1.0 / d


def dec_to_american(d: float | None) -> int | None:
    if not d or d <= 1:
        return None
    if d >= 2:
        return int(round((d - 1) * 100))
    return int(round(-100 / (d - 1)))


def ev_per_1u_risk(p: float | None, dec_odds: float | None) -> float | None:
    """Expected profit in units per 1u risked."""
    if p is None or dec_odds is None or dec_odds <= 1:
        return None
    return p * (dec_odds - 1.0) - (1.0 - p)
