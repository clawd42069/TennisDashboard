#!/usr/bin/env python3
"""Capture today's actionable recommendations into a daily audit log.

This creates the "Actionable bets for the day" section you want, even if you don't bet them.
We snapshot the recommended side/line/price/book at the time of recommendation.

Usage:
  python scripts/capture_daily_actionables.py

Notes:
- We use America/New_York date based on snapshot timestamp.
- Uses the latest snapshot + its actionable candidates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, migrate

ET = ZoneInfo("America/New_York")


def et_date_from_iso(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(ET).date().isoformat()


def main():
    migrate()
    conn = connect()

    snap = conn.execute("SELECT * FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
    if not snap:
        raise SystemExit("No snapshots yet. Call /api/odds first.")

    date_et = et_date_from_iso(snap["ts"])
    snapshot_id = int(snap["id"])

    cands = conn.execute(
        """
        SELECT *
        FROM ranked_candidates
        WHERE snapshot_id = ? AND actionable = 1
        ORDER BY ev_adj DESC NULLS LAST
        """,
        (snapshot_id,),
    ).fetchall()

    created_ts = datetime.now(timezone.utc).isoformat()

    inserted = 0
    for c in cands:
        conn.execute(
            """
            INSERT OR IGNORE INTO daily_actionables (
              date_et, created_ts,
              snapshot_id, candidate_id,
              sport_key, match_id, commence_time, player_a, player_b,
              market_type, side, line, book, price_decimal, price_american,
              confidence, ev, ev_adj,
              result
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                date_et,
                created_ts,
                snapshot_id,
                int(c["id"]),
                snap["sport_key"],
                c["match_id"],
                c["commence_time"],
                c["player_a"],
                c["player_b"],
                c["market_type"],
                c["side"],
                c["line"],
                c["book"],
                c["price_decimal"],
                c["price_american"],
                c["confidence"],
                c["ev"],
                c["ev_adj"],
                "OPEN",
            ),
        )
        inserted += conn.total_changes and 1 or 0

    conn.commit()
    conn.close()

    print(f"Captured daily actionables for {date_et}: {len(cands)} candidates (inserted may include ignores).")


if __name__ == "__main__":
    main()
