#!/usr/bin/env python3
"""Build ATP feature tables used for Lean/Tier outputs.

Creates:
- atp_surface_splits
- atp_recent_oppq_10

Assumes Tennis Abstract ATP import exists in SQLite.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, migrate
from app.elo import build_atp_surface_elo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-year", type=int, default=2015)
    ap.add_argument("--recent-n", type=int, default=10)
    args = ap.parse_args()

    migrate()
    conn = connect()
    cur = conn.cursor()

    # surface splits (simple)
    cur.execute("DROP TABLE IF EXISTS atp_surface_splits")
    cur.execute(
        """
        CREATE TABLE atp_surface_splits AS
        SELECT
          p.player_id as player_id,
          (p.first_name || ' ' || p.last_name) as player_name,
          m.surface as surface,
          COUNT(*) as matches,
          SUM(CASE WHEN m.winner_id = p.player_id THEN 1 ELSE 0 END) as wins,
          SUM(CASE WHEN m.loser_id = p.player_id THEN 1 ELSE 0 END) as losses,
          (1.0 * SUM(CASE WHEN m.winner_id = p.player_id THEN 1 ELSE 0 END) / COUNT(*)) as win_pct
        FROM ta_players p
        JOIN ta_matches m
          ON (m.winner_id = p.player_id OR m.loser_id = p.player_id)
        WHERE m.tourney_date >= ?
          AND m.surface IS NOT NULL
          AND m.surface != ''
        GROUP BY p.player_id, m.surface
        ;
        """,
        (f"{args.since_year}0101",),
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_surface_player ON atp_surface_splits(player_id);")

    # recent opponent quality last N (global, not surface-specific)
    # We'll compute using a window over matches ordered by date per player.
    cur.execute("DROP TABLE IF EXISTS atp_recent_oppq_10")
    cur.execute(
        f"""
        CREATE TABLE atp_recent_oppq_10 AS
        WITH appearances AS (
          SELECT
            tourney_date,
            match_id,
            winner_id as player_id,
            loser_id as opp_id,
            loser_rank as opp_rank,
            1 as is_win
          FROM ta_matches
          WHERE tourney_date >= ?
          UNION ALL
          SELECT
            tourney_date,
            match_id,
            loser_id as player_id,
            winner_id as opp_id,
            winner_rank as opp_rank,
            0 as is_win
          FROM ta_matches
          WHERE tourney_date >= ?
        ),
        ranked AS (
          SELECT
            a.*, 
            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY tourney_date DESC, match_id DESC) as rn
          FROM appearances a
          WHERE opp_rank IS NOT NULL
        )
        SELECT
          player_id,
          COUNT(*) as n,
          AVG(opp_rank) as avg_opp_rank,
          AVG(CASE WHEN rn <= 5 THEN opp_rank ELSE NULL END) as avg_opp_rank_5,
          SUM(is_win) as wins_n,
          (1.0 * SUM(is_win) / COUNT(*)) as win_pct_n
        FROM ranked
        WHERE rn <= {args.recent_n}
        GROUP BY player_id
        ;
        """,
        (f"{args.since_year}0101", f"{args.since_year}0101"),
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oppq_player ON atp_recent_oppq_10(player_id);")

    conn.commit()

    # Build/refresh surface Elo cache (used by candidate engine)
    build_atp_surface_elo(conn, since_year=args.since_year)

    # basic stats
    surf_rows = conn.execute("SELECT COUNT(*) c FROM atp_surface_splits").fetchone()[0]
    oppq_rows = conn.execute("SELECT COUNT(*) c FROM atp_recent_oppq_10").fetchone()[0]
    elo_rows = conn.execute("SELECT COUNT(*) c FROM atp_surface_elo").fetchone()[0]
    conn.close()

    print(f"Built atp_surface_splits rows: {surf_rows}")
    print(f"Built atp_recent_oppq_10 rows: {oppq_rows}")
    print(f"Built atp_surface_elo rows: {elo_rows}")


if __name__ == "__main__":
    main()
