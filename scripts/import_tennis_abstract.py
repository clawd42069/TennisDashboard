#!/usr/bin/env python3
"""Import Tennis Abstract (Jeff Sackmann) data into the dashboard SQLite DB.

MVP focus:
- players
- rankings
- matches (last N years)

Usage:
  python3 scripts/import_tennis_abstract.py --tour atp --since-year 2015
  python3 scripts/import_tennis_abstract.py --tour wta --since-year 2015

Assumes repos are cloned into data/ta_raw/tennis_atp and data/ta_raw/tennis_wta.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, migrate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "ta_raw"


def iter_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def ensure_tables(conn):
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ta_players (
          player_id INTEGER PRIMARY KEY,
          first_name TEXT,
          last_name TEXT,
          hand TEXT,
          birth_date TEXT,
          country_code TEXT,
          height_cm INTEGER
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ta_rankings (
          ranking_date TEXT NOT NULL,
          rank INTEGER NOT NULL,
          player_id INTEGER NOT NULL,
          points INTEGER,
          PRIMARY KEY (ranking_date, rank, player_id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ta_matches (
          match_id TEXT PRIMARY KEY,
          tourney_date TEXT,
          tourney_name TEXT,
          surface TEXT,
          round TEXT,
          winner_id INTEGER,
          loser_id INTEGER,
          winner_rank INTEGER,
          loser_rank INTEGER,
          score TEXT,

          w_ace INTEGER, w_df INTEGER, w_svpt INTEGER, w_1stIn INTEGER, w_1stWon INTEGER,
          w_2ndWon INTEGER, w_SvGms INTEGER, w_bpSaved INTEGER, w_bpFaced INTEGER,
          l_ace INTEGER, l_df INTEGER, l_svpt INTEGER, l_1stIn INTEGER, l_1stWon INTEGER,
          l_2ndWon INTEGER, l_SvGms INTEGER, l_bpSaved INTEGER, l_bpFaced INTEGER
        );
        """
    )

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ta_matches_date ON ta_matches(tourney_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ta_matches_players ON ta_matches(winner_id, loser_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ta_rankings_player ON ta_rankings(player_id, ranking_date);")


def upsert_players(conn, repo_dir: Path):
    player_file = repo_dir / "atp_players.csv"
    if not player_file.exists():
        player_file = repo_dir / "wta_players.csv"
    if not player_file.exists():
        raise FileNotFoundError(f"players csv not found in {repo_dir}")

    rows = list(iter_csv(player_file))
    cur = conn.cursor()
    for r in rows:
        # Modern TA schema uses: name_first,name_last,hand,dob,ioc,height
        pid = int(r["player_id"])
        height = r.get("height")
        height_cm = int(height) if height and str(height).isdigit() else None

        first = r.get("first_name") or r.get("name_first")
        last = r.get("last_name") or r.get("name_last")
        dob = r.get("birth_date") or r.get("dob")
        ioc = r.get("country_code") or r.get("ioc")

        cur.execute(
            """
            INSERT INTO ta_players (player_id, first_name, last_name, hand, birth_date, country_code, height_cm)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
              first_name=excluded.first_name,
              last_name=excluded.last_name,
              hand=excluded.hand,
              birth_date=excluded.birth_date,
              country_code=excluded.country_code,
              height_cm=excluded.height_cm
            """,
            (
                pid,
                first,
                last,
                r.get("hand"),
                dob,
                ioc,
                height_cm,
            ),
        )
    conn.commit()
    return len(rows)


def import_rankings(conn, repo_dir: Path, since_year: int):
    # ranking files are like atp_rankings_00s.csv, atp_rankings_10s.csv, ...
    pattern = re.compile(r"^(atp|wta)_rankings_.*\.csv$")
    files = sorted([p for p in repo_dir.iterdir() if pattern.match(p.name)])
    cur = conn.cursor()
    count = 0
    for f in files:
        for r in iter_csv(f):
            # ranking_date,rank,player_id,points
            d = r.get("ranking_date")
            if not d or int(d[:4]) < since_year:
                continue
            cur.execute(
                """
                INSERT OR IGNORE INTO ta_rankings (ranking_date, rank, player_id, points)
                VALUES (?, ?, ?, ?)
                """,
                (
                    d,
                    int(r.get("rank") or 0),
                    int(r.get("player_id") or 0),
                    int(r.get("points")) if (r.get("points") or "").isdigit() else None,
                ),
            )
            count += 1
    conn.commit()
    return count


def import_matches(conn, repo_dir: Path, since_year: int):
    # match files like atp_matches_2019.csv
    pattern = re.compile(r"^(atp|wta)_matches_(\d{4})\.csv$")
    files = []
    for p in repo_dir.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        year = int(m.group(2))
        if year >= since_year:
            files.append((year, p))
    files.sort()

    cur = conn.cursor()
    inserted = 0

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    cols = [
        "match_id",
        "tourney_date",
        "tourney_name",
        "surface",
        "round",
        "winner_id",
        "loser_id",
        "winner_rank",
        "loser_rank",
        "score",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]

    q = (
        "INSERT OR IGNORE INTO ta_matches ("
        + ",".join(cols)
        + ") VALUES ("
        + ",".join(["?"] * len(cols))
        + ")"
    )

    for year, f in files:
        for r in iter_csv(f):
            # Ensure stable id
            mid = r.get("match_id")
            if not mid:
                # fallback hash
                mid = f"{r.get('tourney_date','')}-{r.get('winner_id','')}-{r.get('loser_id','')}-{r.get('round','')}"

            values = [
                mid,
                r.get("tourney_date"),
                r.get("tourney_name"),
                r.get("surface"),
                r.get("round"),
                to_int(r.get("winner_id")),
                to_int(r.get("loser_id")),
                to_int(r.get("winner_rank")),
                to_int(r.get("loser_rank")),
                r.get("score"),
                to_int(r.get("w_ace")),
                to_int(r.get("w_df")),
                to_int(r.get("w_svpt")),
                to_int(r.get("w_1stIn")),
                to_int(r.get("w_1stWon")),
                to_int(r.get("w_2ndWon")),
                to_int(r.get("w_SvGms")),
                to_int(r.get("w_bpSaved")),
                to_int(r.get("w_bpFaced")),
                to_int(r.get("l_ace")),
                to_int(r.get("l_df")),
                to_int(r.get("l_svpt")),
                to_int(r.get("l_1stIn")),
                to_int(r.get("l_1stWon")),
                to_int(r.get("l_2ndWon")),
                to_int(r.get("l_SvGms")),
                to_int(r.get("l_bpSaved")),
                to_int(r.get("l_bpFaced")),
            ]
            cur.execute(q, values)
            inserted += 1

    conn.commit()
    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", choices=["atp", "wta"], required=True)
    ap.add_argument("--since-year", type=int, default=2015)
    args = ap.parse_args()

    migrate()
    conn = connect()
    ensure_tables(conn)

    repo_dir = RAW / ("tennis_atp" if args.tour == "atp" else "tennis_wta")
    if not repo_dir.exists():
        raise SystemExit(
            f"Repo not found: {repo_dir}. Run scripts/fetch_tennis_abstract.sh first."
        )

    p = upsert_players(conn, repo_dir)
    r = import_rankings(conn, repo_dir, since_year=args.since_year)
    m = import_matches(conn, repo_dir, since_year=args.since_year)

    conn.close()

    print(f"Imported {args.tour.upper()} players: {p}")
    print(f"Imported {args.tour.upper()} rankings rows since {args.since_year}: {r}")
    print(f"Imported {args.tour.upper()} matches since {args.since_year}: {m}")


if __name__ == "__main__":
    main()
