#!/usr/bin/env python3
"""Import Match Charting Project (MCP) aggregate CSVs into SQLite and build per-player style features.

Why MCP first?
- It directly supports Ryan's #1 priority: **style matchups**.
- The repo already contains match-level stat tables (serve basics, rally buckets, net points, etc.)
  so we can ingest and aggregate without writing a full shot-by-shot parser yet.

Current scope (ATP/men files):
- charting-m-stats-Overview.csv
- charting-m-stats-Rally.csv
- charting-m-stats-NetPoints.csv

Output:
- mcp_m_overview / mcp_m_rally / mcp_m_netpoints tables
- style_mcp_player_m aggregated table (per-player style profile)

Usage:
  python scripts/import_mcp_stats.py --root data/style_raw/tennis_MatchChartingProject

Notes:
- MCP coverage is partial and biased (charted matches). Treat as style-signal, not ground truth.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, migrate


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def to_int(x):
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def import_overview(conn, root: Path) -> int:
    path = root / "charting-m-stats-Overview.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cur = conn.cursor()
    n = 0
    for r in iter_csv(path):
        cur.execute(
            """
            INSERT OR REPLACE INTO mcp_m_overview (
              match_id, player, set_label,
              serve_pts, aces, dfs, first_in, first_won, second_in, second_won,
              bk_pts, bp_saved,
              return_pts, return_pts_won,
              winners, winners_fh, winners_bh,
              unforced, unforced_fh, unforced_bh
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                r.get("match_id"),
                r.get("player"),
                r.get("set"),
                to_int(r.get("serve_pts")),
                to_int(r.get("aces")),
                to_int(r.get("dfs")),
                to_int(r.get("first_in")),
                to_int(r.get("first_won")),
                to_int(r.get("second_in")),
                to_int(r.get("second_won")),
                to_int(r.get("bk_pts")),
                to_int(r.get("bp_saved")),
                to_int(r.get("return_pts")),
                to_int(r.get("return_pts_won")),
                to_int(r.get("winners")),
                to_int(r.get("winners_fh")),
                to_int(r.get("winners_bh")),
                to_int(r.get("unforced")),
                to_int(r.get("unforced_fh")),
                to_int(r.get("unforced_bh")),
            ),
        )
        n += 1
    conn.commit()
    return n


def import_rally(conn, root: Path) -> int:
    path = root / "charting-m-stats-Rally.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cur = conn.cursor()
    n = 0
    for r in iter_csv(path):
        cur.execute(
            """
            INSERT OR REPLACE INTO mcp_m_rally (
              match_id, server, returner, row_label,
              pts, pl1_won, pl1_winners, pl1_forced, pl1_unforced,
              pl2_won, pl2_winners, pl2_forced, pl2_unforced
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                r.get("match_id"),
                r.get("server"),
                r.get("returner"),
                r.get("row"),
                to_int(r.get("pts")),
                to_int(r.get("pl1_won")),
                to_int(r.get("pl1_winners")),
                to_int(r.get("pl1_forced")),
                to_int(r.get("pl1_unforced")),
                to_int(r.get("pl2_won")),
                to_int(r.get("pl2_winners")),
                to_int(r.get("pl2_forced")),
                to_int(r.get("pl2_unforced")),
            ),
        )
        n += 1
    conn.commit()
    return n


def import_netpoints(conn, root: Path) -> int:
    path = root / "charting-m-stats-NetPoints.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cur = conn.cursor()
    n = 0
    for r in iter_csv(path):
        cur.execute(
            """
            INSERT OR REPLACE INTO mcp_m_netpoints (
              match_id, player, row_label,
              net_pts, pts_won, net_winner, induced_forced, net_unforced,
              passed_at_net, passing_shot_induced_forced, total_shots
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                r.get("match_id"),
                r.get("player"),
                r.get("row"),
                to_int(r.get("net_pts")),
                to_int(r.get("pts_won")),
                to_int(r.get("net_winner")),
                to_int(r.get("induced_forced")),
                to_int(r.get("net_unforced")),
                to_int(r.get("passed_at_net")),
                to_int(r.get("passing_shot_induced_forced")),
                to_int(r.get("total_shots")),
            ),
        )
        n += 1
    conn.commit()
    return n


def build_style_profiles(conn) -> int:
    """Aggregate MCP match-level stats into a single per-player style row."""
    ts = utc_now_iso()
    cur = conn.cursor()

    # Aggregate from Total rows only
    cur.execute(
        """
        WITH o AS (
          SELECT
            player,
            COUNT(DISTINCT match_id) AS matches,
            SUM(COALESCE(serve_pts,0) + COALESCE(return_pts,0)) AS points,
            SUM(COALESCE(serve_pts,0)) AS serve_pts,
            SUM(COALESCE(return_pts,0)) AS return_pts,
            SUM(COALESCE(aces,0)) AS aces,
            SUM(COALESCE(dfs,0)) AS dfs,
            SUM(COALESCE(first_in,0)) AS first_in,
            SUM(COALESCE(first_won,0)) AS first_won,
            SUM(COALESCE(second_in,0)) AS second_in,
            SUM(COALESCE(second_won,0)) AS second_won,
            SUM(COALESCE(return_pts_won,0)) AS return_pts_won,
            SUM(COALESCE(winners,0)) AS winners,
            SUM(COALESCE(unforced,0)) AS unforced
          FROM mcp_m_overview
          WHERE set_label = 'Total'
          GROUP BY player
        ),
        n AS (
          SELECT
            player,
            SUM(COALESCE(net_pts,0)) AS net_pts,
            SUM(COALESCE(pts_won,0)) AS net_pts_won
          FROM mcp_m_netpoints
          GROUP BY player
        )
        SELECT
          o.player,
          o.matches,
          o.points,
          o.serve_pts,
          o.return_pts,
          CASE WHEN o.serve_pts > 0 THEN (1.0*o.aces/o.serve_pts) END AS ace_rate,
          CASE WHEN o.serve_pts > 0 THEN (1.0*o.dfs/o.serve_pts) END AS df_rate,
          CASE WHEN o.serve_pts > 0 THEN (1.0*o.first_in/o.serve_pts) END AS first_in_pct,
          CASE WHEN o.first_in > 0 THEN (1.0*o.first_won/o.first_in) END AS first_win_pct,
          CASE WHEN o.second_in > 0 THEN (1.0*o.second_won/o.second_in) END AS second_win_pct,
          CASE WHEN o.return_pts > 0 THEN (1.0*o.return_pts_won/o.return_pts) END AS return_win_pct,
          CASE WHEN o.points > 0 THEN (1.0*o.winners/o.points) END AS winner_rate,
          CASE WHEN o.points > 0 THEN (1.0*o.unforced/o.points) END AS unforced_rate,
          CASE WHEN o.points > 0 THEN (1.0*COALESCE(n.net_pts,0)/o.points) END AS net_freq,
          CASE WHEN COALESCE(n.net_pts,0) > 0 THEN (1.0*COALESCE(n.net_pts_won,0)/n.net_pts) END AS net_win_pct
        FROM o
        LEFT JOIN n USING(player)
        ;
        """
    )

    rows = cur.fetchall()

    # Replace style table
    cur.execute("DELETE FROM style_mcp_player_m")
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO style_mcp_player_m (
              player, matches, points, serve_pts, return_pts,
              ace_rate, df_rate, first_in_pct, first_win_pct, second_win_pct,
              return_win_pct, winner_rate, unforced_rate,
              net_freq, net_win_pct,
              updated_ts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                r[0], r[1], r[2], r[3], r[4],
                r[5], r[6], r[7], r[8], r[9],
                r[10], r[11], r[12],
                r[13], r[14],
                ts,
            ),
        )

    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path("data/style_raw/tennis_MatchChartingProject")),
        help="Path to tennis_MatchChartingProject repo",
    )
    args = ap.parse_args()
    root = Path(args.root)

    migrate()
    conn = connect()

    n1 = import_overview(conn, root)
    n2 = import_rally(conn, root)
    n3 = import_netpoints(conn, root)
    n4 = build_style_profiles(conn)

    conn.close()

    print(f"Imported MCP Overview rows: {n1}")
    print(f"Imported MCP Rally rows: {n2}")
    print(f"Imported MCP NetPoints rows: {n3}")
    print(f"Built style_mcp_player_m rows: {n4}")


if __name__ == "__main__":
    main()
