#!/usr/bin/env python3
"""Quick scan: which MCP-derived style dimensions most separate winners vs losers.

Assumption (MCP match_id format): the last two dash-separated tokens are the two players,
where the first of those is the winner and the second is the loser, with underscores
instead of spaces.

This is a *heuristic* but works well enough to rank candidate style dimensions.

Outputs:
- Count of usable matches
- For each feature: mean(winner - loser), std, and win-rate when delta>0

Usage:
  source ../venv/bin/activate
  python analysis/style_dim_scan_mcp.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "tennis.db"


def parse_players(match_id: str):
    parts = (match_id or "").split("-")
    if len(parts) < 2:
        return None, None
    a = parts[-2].replace("_", " ").strip()
    b = parts[-1].replace("_", " ").strip()
    # Some match_ids include extra hyphens inside player strings? very rare.
    return a, b


def main():
    conn = sqlite3.connect(DB_PATH)

    # Total rows only
    df = pd.read_sql_query(
        """
        SELECT match_id, player,
               serve_pts, aces, dfs, first_in, first_won, second_in, second_won,
               return_pts, return_pts_won,
               winners, unforced
        FROM mcp_m_overview
        WHERE set_label = 'Total'
        """,
        conn,
    )

    # Derived per-match per-player features
    df["ace_rate"] = df["aces"] / df["serve_pts"].replace({0: pd.NA})
    df["df_rate"] = df["dfs"] / df["serve_pts"].replace({0: pd.NA})
    df["first_in_pct"] = df["first_in"] / df["serve_pts"].replace({0: pd.NA})
    df["first_win_pct"] = df["first_won"] / df["first_in"].replace({0: pd.NA})
    df["second_win_pct"] = df["second_won"] / df["second_in"].replace({0: pd.NA})
    df["return_win_pct"] = df["return_pts_won"] / df["return_pts"].replace({0: pd.NA})

    df["points"] = (df["serve_pts"].fillna(0) + df["return_pts"].fillna(0)).replace({0: pd.NA})
    df["winner_rate"] = df["winners"] / df["points"]
    df["unforced_rate"] = df["unforced"] / df["points"]

    feat_cols = [
        "ace_rate",
        "df_rate",
        "first_in_pct",
        "first_win_pct",
        "second_win_pct",
        "return_win_pct",
        "winner_rate",
        "unforced_rate",
    ]

    # Map match_id -> (winner, loser)
    parsed = df[["match_id"]].drop_duplicates().copy()
    parsed[["winner", "loser"]] = parsed["match_id"].apply(lambda x: pd.Series(parse_players(x)))

    # Join features for winner/loser
    w = df.merge(parsed[["match_id", "winner"]], left_on=["match_id", "player"], right_on=["match_id", "winner"], how="inner")
    l = df.merge(parsed[["match_id", "loser"]], left_on=["match_id", "player"], right_on=["match_id", "loser"], how="inner")

    w = w[["match_id"] + feat_cols].rename(columns={c: f"w_{c}" for c in feat_cols})
    l = l[["match_id"] + feat_cols].rename(columns={c: f"l_{c}" for c in feat_cols})

    m = w.merge(l, on="match_id", how="inner")

    # Clean NAs
    m = m.dropna()

    out = []
    for c in feat_cols:
        d = m[f"w_{c}"] - m[f"l_{c}"]
        out.append({
            "feature": c,
            "n": int(d.shape[0]),
            "mean_delta": float(d.mean()),
            "std_delta": float(d.std()),
            "p_win_if_delta_pos": float((d > 0).mean()),
        })

    res = pd.DataFrame(out).sort_values("mean_delta", ascending=False)

    print(f"DB: {DB_PATH}")
    print(f"Usable MCP matches (winner+loser feature rows): {m.shape[0]}")
    print("\nTop by mean winner-loser delta:")
    print(res[["feature", "n", "mean_delta", "std_delta", "p_win_if_delta_pos"]].to_string(index=False))


if __name__ == "__main__":
    main()
