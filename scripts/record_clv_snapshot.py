#!/usr/bin/env python3
"""Record a CLV snapshot for current actionable candidates.

Definition (v1):
- For each actionable candidate in the latest snapshot, refetch odds for the sport_key.
- Find the event by match_id (Odds API event id when available; else fallback by teams+commence_time).
- Compute best available price for the candidate side in the candidate market (currently ML/h2h only).
- Store into clv_snapshots with minutes_before_start.

This enables CLV evaluation once we have a "pick time" vs "close time" separation.

Usage:
  python scripts/record_clv_snapshot.py --minutes-before 15

Notes:
- Requires ODDS_API_KEY in env.
- Markets beyond h2h will be added later.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, migrate
from app.odds import get_odds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def minutes_before(start_iso: str | None) -> int | None:
    if not start_iso:
        return None
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    delta = dt - datetime.now(timezone.utc)
    return int(round(delta.total_seconds() / 60.0))


def best_h2h_price(event: dict, side: str) -> float | None:
    best = None
    for bm in (event.get("bookmakers") or []):
        for m in (bm.get("markets") or []):
            if m.get("key") != "h2h":
                continue
            for o in (m.get("outcomes") or []):
                if o.get("name") != side:
                    continue
                p = o.get("price")
                if isinstance(p, (int, float)):
                    best = float(p) if (best is None or float(p) > best) else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes-before", type=int, default=15)
    ap.add_argument("--markets", default="h2h")
    ap.add_argument("--regions", default="us,uk,eu")
    args = ap.parse_args()

    migrate()
    conn = connect()

    snap = conn.execute("SELECT id, sport_key, ts FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
    if not snap:
        raise SystemExit("No snapshots in DB. Call /api/odds first.")

    snapshot_id = int(snap["id"])
    sport_key = snap["sport_key"]

    cands = conn.execute(
        """
        SELECT id, match_id, commence_time, side, market_type
        FROM ranked_candidates
        WHERE snapshot_id = ? AND actionable = 1
        """,
        (snapshot_id,),
    ).fetchall()
    if not cands:
        print("No actionable candidates in latest snapshot.")
        return

    odds, headers = get_odds(sport_key=sport_key, markets=args.markets, regions=args.regions)
    events_by_id = {e.get("id"): e for e in odds if e.get("id")}

    saved = 0
    for c in cands:
        if c["market_type"] != "ML":
            continue

        mid = c["match_id"]
        event = events_by_id.get(mid)

        # Fallback match by teams+commence_time (if match_id isn't Odds API id)
        if event is None:
            ct = c["commence_time"]
            side = c["side"]
            for e in odds:
                if ct and e.get("commence_time") != ct:
                    continue
                if side not in (e.get("home_team"), e.get("away_team")):
                    continue
                event = e
                break

        if event is None:
            continue

        best = best_h2h_price(event, c["side"])
        if best is None:
            continue

        mbs = minutes_before(event.get("commence_time"))
        conn.execute(
            """
            INSERT INTO clv_snapshots (candidate_id, ts, minutes_before_start, best_price_decimal, consensus_price_decimal, best_line, consensus_line, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(c["id"]),
                utc_now_iso(),
                mbs,
                float(best),
                None,
                None,
                None,
                json.dumps({"event": event, "headers": headers}),
            ),
        )
        saved += 1

    conn.commit()
    conn.close()

    print(f"Saved CLV snapshots: {saved} (sport_key={sport_key}, snapshot_id={snapshot_id})")


if __name__ == "__main__":
    main()
