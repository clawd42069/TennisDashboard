#!/usr/bin/env python3
"""Settle (grade) daily actionable recommendations once matches complete.

Approach (v1):
- For OPEN rows in daily_actionables, fetch Odds API scores for that sport_key.
- If the event is completed and we can infer a winner from the score payload,
  mark WIN/LOSS for the recommended side.
- Store raw score_json for audit.

Usage:
  python scripts/settle_daily_actionables.py --days-from 7

Notes:
- Tennis score payload formats can vary. We keep raw score_json regardless.
- Walkovers/retirements: we'll upgrade to VOID logic later if the API exposes it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, migrate
from app.odds import get_scores


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_winner_name(score_event: dict) -> str | None:
    """Try to infer winner name from Odds API score payload."""
    scores = score_event.get("scores") or []
    # Expect two entries like {name: ..., score: ...}
    if len(scores) != 2:
        return None
    try:
        a = scores[0]
        b = scores[1]
        sa = float(a.get("score"))
        sb = float(b.get("score"))
        if sa == sb:
            return None
        return a.get("name") if sa > sb else b.get("name")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-from", type=int, default=7)
    args = ap.parse_args()

    migrate()
    conn = connect()

    # Pull distinct sport_keys that have OPEN actionables
    sport_keys = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT sport_key FROM daily_actionables WHERE result = 'OPEN' AND sport_key IS NOT NULL"
        ).fetchall()
    ]

    total_updated = 0
    for sport_key in sport_keys:
        score_key_candidates = [sport_key]
        if sport_key.startswith("tennis_atp_"):
            score_key_candidates.append("tennis_atp")
        elif sport_key.startswith("tennis_wta_"):
            score_key_candidates.append("tennis_wta")

        events = None
        last_error = None
        for candidate_key in score_key_candidates:
            try:
                events, _headers = get_scores(sport_key=candidate_key, days_from=args.days_from)
                break
            except Exception as e:
                last_error = str(e)

        if events is None:
            print(f"Skipping {sport_key}: {last_error}")
            continue

        by_id = {e.get("id"): e for e in (events or []) if e.get("id")}

        rows = conn.execute(
            """
            SELECT id, match_id, side
            FROM daily_actionables
            WHERE result = 'OPEN' AND sport_key = ?
            """,
            (sport_key,),
        ).fetchall()

        for r in rows:
            event = by_id.get(r["match_id"])
            if not event:
                continue
            if not event.get("completed"):
                continue

            winner = infer_winner_name(event)
            if not winner:
                # Can't infer yet; keep OPEN but store payload
                conn.execute(
                    "UPDATE daily_actionables SET score_json = ? WHERE id = ?",
                    (json.dumps(event), int(r["id"])),
                )
                continue

            result = "WIN" if winner == r["side"] else "LOSS"
            conn.execute(
                """
                UPDATE daily_actionables
                SET result = ?, settled_ts = ?, score_json = ?
                WHERE id = ?
                """,
                (result, utc_now_iso(), json.dumps(event), int(r["id"])),
            )
            total_updated += 1

    conn.commit()
    conn.close()

    print(f"Settled daily actionables updated: {total_updated}")


if __name__ == "__main__":
    main()
