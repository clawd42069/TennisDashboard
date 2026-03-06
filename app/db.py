import os
import sqlite3
from pathlib import Path

# In cloud deployments, set TENNIS_DB_PATH to a persistent disk path (e.g. /var/data/tennis.db)
# If that path isn't writable (disk not mounted yet), we fall back to a local ./data/tennis.db so deploy can succeed.
_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "tennis.db"
DB_PATH = Path(os.getenv("TENNIS_DB_PATH") or _DEFAULT_DB)


def connect():
    global DB_PATH
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fall back if persistent disk path isn't mounted / writable yet
        DB_PATH = _DEFAULT_DB
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def migrate():
    conn = connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          sport_key TEXT NOT NULL,
          match_id TEXT NOT NULL,
          payload_json TEXT NOT NULL
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_ts ON odds_snapshots(ts);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(match_id);"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_bets (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          match_id TEXT,
          match_label TEXT,
          tournament TEXT,
          player TEXT NOT NULL,
          market TEXT NOT NULL,
          odds_decimal REAL,
          odds_american INTEGER,
          units REAL NOT NULL,
          note TEXT,
          result TEXT,           -- WIN/LOSS/PUSH/OPEN
          pnl_units REAL,        -- profit/loss in units
          pnl_dollars REAL,      -- profit/loss in dollars
          settled_ts TEXT
        );
        """
    )

    # Forward-compatible ALTERs for existing DBs
    try:
        cur.execute("ALTER TABLE paper_bets ADD COLUMN match_label TEXT;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE paper_bets ADD COLUMN odds_american INTEGER;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE paper_bets ADD COLUMN pnl_units REAL;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE paper_bets ADD COLUMN pnl_dollars REAL;")
    except Exception:
        pass

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategies (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          description TEXT,
          rules_json TEXT,
          created_ts TEXT NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()
