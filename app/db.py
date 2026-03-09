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
    """Create/upgrade DB schema.

    Notes:
    - Keep early tables (odds_snapshots/paper_bets) for backward compatibility.
    - New v2 schema adds normalized refresh snapshots + ranked candidates so we can
      compute EV/CLV and drive Actionable vs Debug views.
    """
    conn = connect()
    cur = conn.cursor()

    # ---------------- Legacy odds snapshots (raw JSON per match) ----------------
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_odds_ts ON odds_snapshots(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(match_id);")

    # ---------------- V2 snapshots (one row per refresh) ----------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          tour TEXT NOT NULL,              -- ATP
          sport_key TEXT NOT NULL,
          markets TEXT NOT NULL,           -- comma list
          regions TEXT NOT NULL,           -- comma list
          top_n INTEGER NOT NULL,
          refresh_interval_sec INTEGER NOT NULL,
          model_version TEXT,
          slate_json TEXT                  -- optional high-level context
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);")

    # ---------------- V2 ranked candidates (Top N per snapshot) ----------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ranked_candidates (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          snapshot_id INTEGER NOT NULL,
          created_at TEXT NOT NULL,

          match_id TEXT NOT NULL,
          commence_time TEXT,
          tournament TEXT,
          surface TEXT,
          player_a TEXT,
          player_b TEXT,

          market_type TEXT NOT NULL,       -- ML|SPREAD|TOTAL
          side TEXT NOT NULL,              -- e.g. player name, OVER, UNDER
          line REAL,                       -- spread/total line, null for ML
          price_decimal REAL,
          price_american INTEGER,
          book TEXT,

          p0 REAL,                         -- baseline prob
          p_final REAL,                    -- final prob
          q_implied REAL,                  -- implied prob from price
          ev REAL,                         -- expected value per 1u risk
          confidence REAL,                 -- 0..1
          ev_adj REAL,                     -- ev*confidence

          delta_elo_surface REAL,
          delta_sr REAL,
          delta_recency REAL,
          delta_z_raw REAL,
          delta_z_capped REAL,
          matchup_flags_json TEXT,

          view_mode TEXT NOT NULL,         -- actionable|debug
          actionable INTEGER NOT NULL DEFAULT 0,
          units_suggested REAL,

          FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_snapshot ON ranked_candidates(snapshot_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_match ON ranked_candidates(match_id);")

    # ---------------- Surface Elo cache (ATP) ----------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS atp_surface_elo (
          player_id INTEGER NOT NULL,
          surface TEXT NOT NULL,
          elo REAL NOT NULL,
          matches INTEGER NOT NULL,
          updated_ts TEXT NOT NULL,
          PRIMARY KEY (player_id, surface)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_atp_surface_elo_surface ON atp_surface_elo(surface);")

    # ---------------- CLV snapshots (for actionable picks) ----------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clv_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          candidate_id INTEGER NOT NULL,
          ts TEXT NOT NULL,
          minutes_before_start INTEGER,
          best_price_decimal REAL,
          consensus_price_decimal REAL,
          best_line REAL,
          consensus_line REAL,
          payload_json TEXT,
          FOREIGN KEY(candidate_id) REFERENCES ranked_candidates(id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_clv_candidate ON clv_snapshots(candidate_id);")

    # ---------------- Match Charting Project (MCP) style stats (ATP/WTA) ----------------

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_m_overview (
          match_id TEXT NOT NULL,
          player TEXT NOT NULL,
          set_label TEXT NOT NULL,
          serve_pts INTEGER,
          aces INTEGER,
          dfs INTEGER,
          first_in INTEGER,
          first_won INTEGER,
          second_in INTEGER,
          second_won INTEGER,
          bk_pts INTEGER,
          bp_saved INTEGER,
          return_pts INTEGER,
          return_pts_won INTEGER,
          winners INTEGER,
          winners_fh INTEGER,
          winners_bh INTEGER,
          unforced INTEGER,
          unforced_fh INTEGER,
          unforced_bh INTEGER,
          PRIMARY KEY (match_id, player, set_label)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_m_rally (
          match_id TEXT NOT NULL,
          server TEXT NOT NULL,
          returner TEXT NOT NULL,
          row_label TEXT NOT NULL,
          pts INTEGER,
          pl1_won INTEGER,
          pl1_winners INTEGER,
          pl1_forced INTEGER,
          pl1_unforced INTEGER,
          pl2_won INTEGER,
          pl2_winners INTEGER,
          pl2_forced INTEGER,
          pl2_unforced INTEGER,
          PRIMARY KEY (match_id, server, returner, row_label)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_m_netpoints (
          match_id TEXT NOT NULL,
          player TEXT NOT NULL,
          row_label TEXT NOT NULL,
          net_pts INTEGER,
          pts_won INTEGER,
          net_winner INTEGER,
          induced_forced INTEGER,
          net_unforced INTEGER,
          passed_at_net INTEGER,
          passing_shot_induced_forced INTEGER,
          total_shots INTEGER,
          PRIMARY KEY (match_id, player, row_label)
        );
        """
    )

    # Aggregated style features per player (men), derived from MCP
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS style_mcp_player_m (
          player TEXT PRIMARY KEY,
          matches INTEGER,
          points INTEGER,
          serve_pts INTEGER,
          return_pts INTEGER,
          ace_rate REAL,
          df_rate REAL,
          first_in_pct REAL,
          first_win_pct REAL,
          second_win_pct REAL,
          return_win_pct REAL,
          winner_rate REAL,
          unforced_rate REAL,
          net_freq REAL,
          net_win_pct REAL,
          updated_ts TEXT NOT NULL
        );
        """
    )

    # ---------------- Daily actionable recommendations (audit log) ----------------

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_actionables (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          date_et TEXT NOT NULL,            -- YYYY-MM-DD in America/New_York
          created_ts TEXT NOT NULL,

          snapshot_id INTEGER,
          candidate_id INTEGER,

          sport_key TEXT,
          match_id TEXT,
          commence_time TEXT,
          player_a TEXT,
          player_b TEXT,

          market_type TEXT NOT NULL,
          side TEXT NOT NULL,
          line REAL,
          book TEXT,
          price_decimal REAL,
          price_american INTEGER,

          confidence REAL,
          ev REAL,
          ev_adj REAL,

          close_price_decimal REAL,
          close_price_american INTEGER,
          close_ts TEXT,

          result TEXT,                      -- WIN|LOSS|PUSH|VOID|OPEN
          settled_ts TEXT,
          score_json TEXT,

          UNIQUE(date_et, candidate_id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_actionables_date ON daily_actionables(date_et);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_actionables_match ON daily_actionables(match_id);")

    # ---------------- Paper bets (existing) ----------------
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
          result TEXT,           -- WIN/LOSS/PUSH/OPEN/VOID
          pnl_units REAL,        -- profit/loss in units
          pnl_dollars REAL,      -- profit/loss in dollars
          settled_ts TEXT,
          source_candidate_id INTEGER
        );
        """
    )

    # Forward-compatible ALTERs for existing DBs
    for stmt in [
        "ALTER TABLE paper_bets ADD COLUMN match_label TEXT;",
        "ALTER TABLE paper_bets ADD COLUMN odds_american INTEGER;",
        "ALTER TABLE paper_bets ADD COLUMN pnl_units REAL;",
        "ALTER TABLE paper_bets ADD COLUMN pnl_dollars REAL;",
        "ALTER TABLE paper_bets ADD COLUMN source_candidate_id INTEGER;",
    ]:
        try:
            cur.execute(stmt)
        except Exception:
            pass

    # ---------------- Strategies (existing) ----------------
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
