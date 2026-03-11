from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from types import SimpleNamespace

from flask import Flask, render_template, request, redirect, url_for, jsonify
import threading
import subprocess
import sys

import math

from .db import migrate, connect
from .odds import list_sports, get_odds, get_scores, normalize_match
from .ratings import rate_match

from .engine import generate_ml_candidates
from .modeling import dec_to_american

import os

APP_PORT = int(os.getenv("PORT", "8008"))


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def create_app():
    app = Flask(__name__)
    app.jinja_env.globals['env'] = os.environ
    migrate()

    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
    _bootstrap = {"running": False, "status": "idle", "last": None}
    _settler = {"running": False, "last": None, "last_result": None, "last_on_demand": None}

    def _auth_ok():
        if not ADMIN_TOKEN:
            return False
        token = request.args.get("token") or request.headers.get("Authorization")
        return token == ADMIN_TOKEN

    def _run_bootstrap():
        _bootstrap["running"] = True
        _bootstrap["status"] = "running"
        _bootstrap["last"] = utc_now_iso()
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            py = sys.executable
            # Fetch raw Tennis Abstract repos (idempotent) then import ATP + build features (includes surface Elo build)
            subprocess.check_call(["bash", os.path.join(root, "scripts", "fetch_tennis_abstract.sh")])
            subprocess.check_call([py, os.path.join(root, "scripts", "import_tennis_abstract.py"), "--tour", "atp", "--since-year", "2015"])
            subprocess.check_call([py, os.path.join(root, "scripts", "build_features_atp.py"), "--since-year", "2015", "--recent-n", "10"])
            _bootstrap["status"] = "ok"
        except Exception as e:
            _bootstrap["status"] = f"error: {e}"
        finally:
            _bootstrap["running"] = False
            _bootstrap["last"] = utc_now_iso()

    @app.get("/")
    def index():
        return redirect(url_for("matchups"))

    @app.get("/matchups")
    def matchups():
        """Tab B: Matchup report (raw odds + outputs)."""
        return render_template("matchups.html")

    ET = ZoneInfo("America/New_York")
    MATCH_LIVE_WINDOW_MIN = int(os.getenv("MATCH_LIVE_WINDOW_MIN") or 180)

    def fmt_et(iso: str | None):
        if not iso:
            return ""
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            return iso
        dt_et = dt.astimezone(ET)
        hour = dt_et.hour
        minute = dt_et.minute
        ampm = "PM" if hour >= 12 else "AM"
        hour12 = hour % 12
        if hour12 == 0:
            hour12 = 12
        time_str = f"{hour12}:{minute:02d} {ampm}"
        date_str = dt_et.strftime("%m/%d/%y")
        return f"{time_str} {date_str}"

    app.jinja_env.filters["fmt_et"] = fmt_et

    def _backfill_paper_start_fields(conn):
        rows = conn.execute(
            "SELECT id, match_id, commence_time, start_date_et FROM paper_bets WHERE commence_time IS NULL OR start_date_et IS NULL"
        ).fetchall()
        for row in rows:
            commence_time = row["commence_time"]
            start_date_et = row["start_date_et"]
            if row["match_id"] and (not commence_time or not start_date_et):
                snap = conn.execute(
                    "SELECT payload_json FROM odds_snapshots WHERE match_id = ? ORDER BY id DESC LIMIT 1",
                    (row["match_id"],),
                ).fetchone()
                if snap:
                    try:
                        payload = json.loads(snap["payload_json"] or '{}')
                        commence_time = commence_time or payload.get("commence_time")
                    except Exception:
                        pass
            start_date_et = start_date_et or _et_date_from_iso(commence_time)
            conn.execute(
                "UPDATE paper_bets SET commence_time = COALESCE(commence_time, ?), start_date_et = COALESCE(start_date_et, ?) WHERE id = ?",
                (commence_time, start_date_et, int(row["id"])),
            )

    def _paper_state(conn):
        _backfill_paper_start_fields(conn)
        conn.commit()
        bets = conn.execute(
            "SELECT * FROM paper_bets ORDER BY id DESC LIMIT 200"
        ).fetchall()

        latest = conn.execute("SELECT ts FROM odds_snapshots ORDER BY ts DESC LIMIT 1").fetchone()
        match_labels = {}
        if latest:
            rows = conn.execute(
                "SELECT match_id, payload_json FROM odds_snapshots WHERE ts = ?",
                (latest["ts"],),
            ).fetchall()
            for r in rows:
                try:
                    m = json.loads(r["payload_json"])
                    match_labels[r["match_id"]] = f"{m.get('away_team','')} vs {m.get('home_team','')}"
                except Exception:
                    pass

        enriched = []
        for b in bets:
            d = dict(b)
            if not d.get("match_label") and d.get("match_id"):
                d["match_label"] = match_labels.get(d.get("match_id"))
            enriched.append(d)

        total_bets = len(enriched)
        settled = [b for b in enriched if (b.get("result") in ("WIN", "LOSS", "PUSH"))]
        wins = [b for b in settled if b.get("result") == "WIN"]
        losses = [b for b in settled if b.get("result") == "LOSS"]
        open_bets = [b for b in enriched if (b.get("result") or "OPEN") == "OPEN"]

        total_pnl = sum((b.get("pnl_dollars") or 0.0) for b in settled)
        total_units = sum((b.get("units") or 0.0) for b in enriched)
        win_rate = (len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) > 0 else None

        by_type = {}
        for b in enriched:
            t = b.get("market") or ""
            by_type.setdefault(t, {"bets": 0, "wins": 0, "losses": 0, "pnl": 0.0})
            by_type[t]["bets"] += 1
            if b.get("result") == "WIN":
                by_type[t]["wins"] += 1
            elif b.get("result") == "LOSS":
                by_type[t]["losses"] += 1
            if b.get("result") in ("WIN", "LOSS", "PUSH"):
                by_type[t]["pnl"] += float(b.get("pnl_dollars") or 0.0)

        perf = {
            "total_pnl": total_pnl,
            "wins": len(wins),
            "losses": len(losses),
            "settled": len(settled),
            "total_bets": total_bets,
            "open_bets": len(open_bets),
            "win_rate": win_rate,
            "total_units": total_units,
            "by_type": by_type,
        }
        return {"bets": enriched, "perf": perf}

    @app.get("/paper")
    def paper():
        """Tab E: paper trading tracker."""
        conn = connect()
        state = _paper_state(conn)
        conn.close()
        return render_template("paper.html", bets=state["bets"], perf=state["perf"])

    def dec_to_american(dec: float | None):
        if not dec or dec <= 1:
            return None
        if dec >= 2:
            return int(round((dec - 1) * 100))
        # negative odds
        return int(round(-100 / (dec - 1)))

    UNIT_DOLLARS = float((request.args.get("unit_dollars") if False else 0) or 0)  # unused

    def pnl_units_for_result(result: str, units: float, american: int | None):
        if result == "WIN":
            if american is None:
                return None
            if american > 0:
                return units * (american / 100.0)
            # american negative
            return units * (100.0 / abs(american))
        if result == "LOSS":
            return -units
        if result == "PUSH":
            return 0.0
        return None

    def units_to_dollars(pnl_units: float | None, unit_value: float = 500.0):
        if pnl_units is None:
            return None
        return pnl_units * unit_value

    def _insert_paper_bet(conn, payload: dict):
        odds_dec = payload.get("odds_decimal")
        odds_dec_f = float(odds_dec) if odds_dec not in (None, "") else None
        commence_time = payload.get("commence_time") or None
        start_date_et = payload.get("start_date_et") or _et_date_from_iso(commence_time)
        odds_am = payload.get("odds_american")
        odds_am_i = int(odds_am) if odds_am not in (None, "") else None
        if odds_am_i is None and odds_dec_f is not None:
            odds_am_i = dec_to_american(odds_dec_f)

        units = float(payload.get("units") or 0)
        result = payload.get("result") or "OPEN"
        pnl_units = pnl_units_for_result(result, units, odds_am_i)
        pnl_dollars = units_to_dollars(pnl_units, 500.0)
        ts = utc_now_iso()

        cur = conn.execute(
            """
            INSERT INTO paper_bets (ts, match_id, match_label, tournament, player, market, odds_decimal, odds_american, units, note, result, pnl_units, pnl_dollars, settled_ts, source_candidate_id, commence_time, start_date_et)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                payload.get("match_id") or None,
                payload.get("match_label") or None,
                payload.get("tournament") or None,
                payload.get("player") or "",
                payload.get("market") or "h2h",
                odds_dec_f,
                odds_am_i,
                units,
                payload.get("note") or None,
                result,
                pnl_units,
                pnl_dollars,
                ts if result in ("WIN", "LOSS", "PUSH") else None,
                payload.get("source_candidate_id") or None,
                commence_time,
                start_date_et,
            ),
        )
        row_id = cur.lastrowid
        row = conn.execute("SELECT * FROM paper_bets WHERE id = ?", (row_id,)).fetchone()
        return dict(row) if row else {"id": row_id}

    @app.post("/paper/add")
    def paper_add():
        conn = connect()
        _insert_paper_bet(conn, dict(request.form))
        conn.commit()
        conn.close()
        return redirect(url_for("paper"))

    @app.post("/api/paper/add")
    def api_paper_add():
        payload = request.get_json(silent=True) or {}
        if not payload.get("player"):
            return jsonify({"ok": False, "error": "player required"}), 400
        if not payload.get("units"):
            return jsonify({"ok": False, "error": "units required"}), 400
        conn = connect()
        row = _insert_paper_bet(conn, payload)
        conn.commit()
        conn.close()
        return jsonify({"ok": True, "bet": row})

    @app.post("/api/paper/settle")
    def api_paper_settle():
        payload = request.get_json(silent=True) or {}
        bet_id = payload.get("bet_id")
        result = (payload.get("result") or "").upper()
        if not bet_id:
            return jsonify({"ok": False, "error": "bet_id required"}), 400
        if result not in ("WIN", "LOSS", "PUSH", "OPEN"):
            return jsonify({"ok": False, "error": "invalid result"}), 400
        conn = connect()
        row = conn.execute("SELECT * FROM paper_bets WHERE id = ?", (int(bet_id),)).fetchone()
        if not row:
            conn.close()
            return jsonify({"ok": False, "error": "bet not found"}), 404
        units = float(row["units"] or 0.0)
        odds_am = row["odds_american"]
        pnl_units = pnl_units_for_result(result, units, odds_am)
        pnl_dollars = units_to_dollars(pnl_units, 500.0)
        settled_ts = utc_now_iso() if result in ("WIN", "LOSS", "PUSH") else None
        conn.execute(
            "UPDATE paper_bets SET result = ?, pnl_units = ?, pnl_dollars = ?, settled_ts = ? WHERE id = ?",
            (result, pnl_units, pnl_dollars, settled_ts, int(bet_id)),
        )
        conn.commit()
        out = conn.execute("SELECT * FROM paper_bets WHERE id = ?", (int(bet_id),)).fetchone()
        conn.close()
        return jsonify({"ok": True, "bet": dict(out) if out else {"id": bet_id}})

    @app.get("/strategies")
    def strategies():
        """Tab D: strategies + backtests + current selection audit."""
        conn = connect()
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY id DESC LIMIT 200"
        ).fetchall()
        audit = _strategy_audit_summary(conn)
        conn.close()
        return render_template("strategies.html", strategies=rows, audit=audit)

    @app.get("/player")
    def player():
        """Tab C: player profile (stub)."""
        return render_template("player.html")

    # ---------- API ----------

    @app.get("/health")
    def health():
        return jsonify({
            "ok": True,
            "commit": os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT"),
            "db_path": os.getenv("TENNIS_DB_PATH"),
        })

    @app.get("/api/tennis_sports")
    def api_tennis_sports():
        sports, headers = list_sports()
        tennis = [s for s in sports if (s.get("group") == "Tennis") or ("tennis" in (s.get("key") or ""))]
        return jsonify({
            "tennis": tennis,
            "requests_remaining": headers.get("x-requests-remaining"),
            "requests_used": headers.get("x-requests-used"),
        })

    # ---------- Admin bootstrap (historical import + feature build) ----------

    @app.get("/admin/bootstrap/status")
    def admin_bootstrap_status():
        if not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return jsonify({"ok": True, **_bootstrap})

    @app.post("/admin/bootstrap/run")
    def admin_bootstrap_run():
        if not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        if _bootstrap.get("running"):
            return jsonify({"ok": True, **_bootstrap})
        t = threading.Thread(target=_run_bootstrap, daemon=True)
        t.start()
        return jsonify({"ok": True, **_bootstrap})

    # ---------- Admin: settle daily actionables ----------

    def _infer_winner_name(score_event: dict):
        scores = score_event.get("scores") or []
        if len(scores) != 2:
            return None
        try:
            a, b = scores[0], scores[1]
            sa, sb = float(a.get("score")), float(b.get("score"))
            if sa == sb:
                return None
            return a.get("name") if sa > sb else b.get("name")
        except Exception:
            return None

    def _archive_score_events(conn, source: str, sport_key: str, events: list[dict]):
        fetched_ts = utc_now_iso()
        for event in (events or []):
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO score_event_archive (
                      source, sport_key, event_id, home_team, away_team, commence_time,
                      completed, winner_name, scores_json, last_update, fetched_ts, payload_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        source,
                        sport_key,
                        event.get("id"),
                        event.get("home_team"),
                        event.get("away_team"),
                        event.get("commence_time"),
                        1 if event.get("completed") else 0,
                        _infer_winner_name(event),
                        json.dumps(event.get("scores")) if event.get("scores") is not None else None,
                        event.get("last_update"),
                        fetched_ts,
                        json.dumps(event),
                    ),
                )
            except Exception:
                pass

    def _load_archived_result(conn, match_id: str | None, player_a: str | None, player_b: str | None, commence_time: str | None):
        if match_id:
            row = conn.execute(
                """
                SELECT winner_name, payload_json FROM score_event_archive
                WHERE event_id = ? AND completed = 1 AND winner_name IS NOT NULL
                ORDER BY fetched_ts DESC LIMIT 1
                """,
                (match_id,),
            ).fetchone()
            if row:
                return dict(row)
        if player_a and player_b:
            row = conn.execute(
                """
                SELECT winner_name, payload_json FROM score_event_archive
                WHERE completed = 1 AND winner_name IS NOT NULL
                  AND ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
                ORDER BY fetched_ts DESC LIMIT 1
                """,
                (player_a, player_b, player_b, player_a),
            ).fetchone()
            if row:
                return dict(row)
        return None

    def _settle_open_daily_rows(conn, table_name: str, date_et: str | None = None, days_from: int = 1):
        """Settle OPEN daily rows (actionables/watchlist) from score feed."""
        days_from = max(1, min(int(days_from or 1), 1))

        sql = f"SELECT DISTINCT sport_key FROM {table_name} WHERE result = 'OPEN' AND sport_key IS NOT NULL"
        params = []
        if date_et:
            sql += " AND date_et = ?"
            params.append(date_et)
        sk_rows = conn.execute(sql, tuple(params)).fetchall()
        sport_keys = [r[0] for r in sk_rows]

        updated = 0
        errors = []
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
                    events, _headers = get_scores(sport_key=candidate_key, days_from=days_from)
                    break
                except Exception as e:
                    last_error = str(e)

            if events is None:
                errors.append({"sport_key": sport_key, "error": last_error})
                continue

            _archive_score_events(conn, "odds_api", sport_key, events or [])
            by_id = {e.get("id"): e for e in (events or []) if e.get("id")}
            row_sql = f"SELECT id, match_id, side, player_a, player_b, commence_time FROM {table_name} WHERE result = 'OPEN' AND sport_key = ?"
            row_params = [sport_key]
            if date_et:
                row_sql += " AND date_et = ?"
                row_params.append(date_et)
            rows = conn.execute(row_sql, tuple(row_params)).fetchall()

            for r in rows:
                event = by_id.get(r["match_id"])
                winner = None
                payload_json = None
                if event and event.get("completed"):
                    winner = _infer_winner_name(event)
                    payload_json = json.dumps(event)
                else:
                    archived = _load_archived_result(conn, r["match_id"], r["player_a"], r["player_b"], r["commence_time"])
                    if archived:
                        winner = archived.get("winner_name")
                        payload_json = archived.get("payload_json")
                if not winner:
                    continue
                result = "WIN" if winner == r["side"] else "LOSS"
                conn.execute(
                    f"UPDATE {table_name} SET result = ?, settled_ts = ?, score_json = ? WHERE id = ?",
                    (result, utc_now_iso(), payload_json, int(r["id"])),
                )
                updated += 1

        return {"updated": updated, "errors": errors}

    def _settle_open_actionables(conn, date_et: str | None = None, days_from: int = 1):
        return _settle_open_daily_rows(conn, "daily_actionables", date_et=date_et, days_from=days_from)

    def _settle_open_watchlist(conn, date_et: str | None = None, days_from: int = 1):
        return _settle_open_daily_rows(conn, "daily_watchlist", date_et=date_et, days_from=days_from)

    def _should_run_on_demand_settle(min_interval_sec: int = 60) -> bool:
        last = _settler.get("last_on_demand")
        if not last:
            return True
        try:
            dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) - dt >= timedelta(seconds=min_interval_sec)
        except Exception:
            return True

    def _settle_open_paper_bets(conn, days_from: int = 1):
        """Settle OPEN paper bets when the final match winner is available.

        Current auto-grading supports h2h bets only. Spreads/totals remain manual until
        we add richer score parsing / line logic.
        """
        days_from = max(1, min(int(days_from or 1), 1))
        sk_rows = conn.execute(
            """
            SELECT DISTINCT os.sport_key
            FROM paper_bets pb
            JOIN odds_snapshots os ON os.match_id = pb.match_id
            WHERE COALESCE(pb.result, 'OPEN') = 'OPEN'
              AND pb.match_id IS NOT NULL
              AND os.sport_key IS NOT NULL
            """
        ).fetchall()
        sport_keys = [r[0] for r in sk_rows]

        updated = 0
        errors = []
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
                    events, _headers = get_scores(sport_key=candidate_key, days_from=days_from)
                    break
                except Exception as e:
                    last_error = str(e)

            if events is None:
                errors.append({"sport_key": sport_key, "error": last_error})
                continue

            _archive_score_events(conn, "odds_api", sport_key, events or [])
            by_id = {e.get("id"): e for e in (events or []) if e.get("id")}
            rows = conn.execute(
                """
                SELECT DISTINCT pb.id, pb.match_id, pb.player, pb.market, pb.units, pb.odds_american, os.payload_json
                FROM paper_bets pb
                JOIN odds_snapshots os ON os.match_id = pb.match_id
                WHERE COALESCE(pb.result, 'OPEN') = 'OPEN'
                  AND os.sport_key = ?
                """,
                (sport_key,),
            ).fetchall()

            for r in rows:
                event = by_id.get(r["match_id"])
                market = (r["market"] or "").lower()
                if market != "h2h":
                    continue
                winner = None
                if event and event.get("completed"):
                    winner = _infer_winner_name(event)
                else:
                    player_a = player_b = None
                    try:
                        payload = json.loads(r["payload_json"] or '{}')
                        player_a = payload.get('away_team')
                        player_b = payload.get('home_team')
                    except Exception:
                        pass
                    archived = _load_archived_result(conn, r["match_id"], player_a, player_b, None)
                    if archived:
                        winner = archived.get("winner_name")
                if not winner:
                    continue
                result = "WIN" if winner == r["player"] else "LOSS"
                pnl_units = pnl_units_for_result(result, float(r["units"] or 0.0), r["odds_american"])
                pnl_dollars = units_to_dollars(pnl_units, 500.0)
                conn.execute(
                    "UPDATE paper_bets SET result = ?, pnl_units = ?, pnl_dollars = ?, settled_ts = ? WHERE id = ?",
                    (result, pnl_units, pnl_dollars, utc_now_iso(), int(r["id"])),
                )
                updated += 1

        return {"updated": updated, "errors": errors}

    def _background_settler_loop():
        interval_sec = max(300, int(os.getenv("ACTIONABLES_SETTLER_INTERVAL_SEC") or 300))
        while True:
            try:
                conn = connect()
                open_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM daily_actionables WHERE result = 'OPEN'"
                ).fetchone()
                open_n = int((open_row or {"n": 0})["n"])
                paper_open_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM paper_bets WHERE COALESCE(result, 'OPEN') = 'OPEN'"
                ).fetchone()
                paper_open_n = int((paper_open_row or {"n": 0})["n"])
                watchlist_open_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM daily_watchlist WHERE result = 'OPEN'"
                ).fetchone()
                watchlist_open_n = int((watchlist_open_row or {"n": 0})["n"])
                if open_n > 0 or watchlist_open_n > 0 or paper_open_n > 0:
                    settle = _settle_open_actionables(conn, days_from=1) if open_n > 0 else {"updated": 0, "errors": []}
                    watchlist_settle = _settle_open_watchlist(conn, days_from=1) if watchlist_open_n > 0 else {"updated": 0, "errors": []}
                    paper_settle = _settle_open_paper_bets(conn, days_from=1) if paper_open_n > 0 else {"updated": 0, "errors": []}
                    conn.commit()
                    _settler["last_result"] = {"open": open_n, "watchlist_open": watchlist_open_n, "paper_open": paper_open_n, "actionables": settle, "watchlist": watchlist_settle, "paper": paper_settle}
                else:
                    _settler["last_result"] = {"open": 0, "watchlist_open": 0, "paper_open": 0, "updated": 0, "errors": []}
                _settler["last"] = utc_now_iso()
                conn.close()
            except Exception as e:
                _settler["last"] = utc_now_iso()
                _settler["last_result"] = {"error": str(e)}
            finally:
                import time
                time.sleep(interval_sec)

    if os.getenv("ACTIONABLES_SETTLER_ENABLED", "1") == "1" and not _settler["running"]:
        threading.Thread(target=_background_settler_loop, daemon=True).start()
        _settler["running"] = True

    @app.get("/admin/odds/sports")
    def admin_odds_sports():
        if not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        try:
            sports, _h = list_sports()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        # return tennis-related keys first to keep payload small
        tennis = [s for s in (sports or []) if str(s.get("key", "")).startswith("tennis")]
        return jsonify({"ok": True, "tennis": tennis, "count_tennis": len(tennis), "count_all": len(sports or [])})

    @app.get("/admin/actionables/settler_status")
    def admin_actionables_settler_status():
        if not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return jsonify({"ok": True, **_settler})

    @app.get("/admin/actionables/debug")
    def admin_actionables_debug():
        if not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        conn = connect()
        by_res = conn.execute(
            "SELECT result, COUNT(*) AS n FROM daily_actionables GROUP BY result ORDER BY result"
        ).fetchall()
        by_sk = conn.execute(
            "SELECT sport_key, result, COUNT(*) AS n FROM daily_actionables GROUP BY sport_key, result ORDER BY sport_key, result"
        ).fetchall()
        sample = conn.execute(
            "SELECT id, date_et, sport_key, result, match_id, side FROM daily_actionables ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return jsonify(
            {
                "ok": True,
                "by_result": [dict(r) for r in by_res],
                "by_sport_key": [dict(r) for r in by_sk],
                "sample": [dict(r) for r in sample],
            }
        )

    @app.post("/admin/actionables/settle")
    def admin_actionables_settle():
        if not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401

        days_from = int(request.args.get("days_from") or 1)
        conn = connect()

        sk_rows = conn.execute(
            "SELECT DISTINCT sport_key FROM daily_actionables WHERE result = 'OPEN' AND sport_key IS NOT NULL"
        ).fetchall()
        sport_keys = [r[0] for r in sk_rows]
        open_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM daily_actionables WHERE result = 'OPEN'"
        ).fetchone()
        settle = _settle_open_actionables(conn, days_from=days_from)

        conn.commit()
        conn.close()

        return jsonify({"ok": True, "sport_keys": sport_keys, "checked": int((open_rows or {"n": 0})["n"]), "updated": settle["updated"], "errors": settle["errors"]})

    @app.get("/api/paper/state")
    def api_paper_state():
        conn = connect()
        open_row = conn.execute(
            "SELECT COUNT(*) AS n FROM paper_bets WHERE COALESCE(result, 'OPEN') = 'OPEN'"
        ).fetchone()
        open_n = int((open_row or {"n": 0})["n"])
        settle_info = None
        if open_n > 0 and _should_run_on_demand_settle(60):
            settle_info = _settle_open_paper_bets(conn, days_from=1)
            conn.commit()
            _settler["last_on_demand"] = utc_now_iso()
        state = _paper_state(conn)
        conn.close()
        if settle_info is not None:
            state["settle"] = settle_info
        return jsonify(state)

    @app.get("/api/paper_candidates")
    def api_paper_candidates():
        """Return a compact list of upcoming/live matches from most recent odds snapshot(s).

        Used to populate dropdowns on the Paper Tracker.
        """
        conn = connect()
        row = conn.execute("SELECT ts FROM odds_snapshots ORDER BY ts DESC LIMIT 1").fetchone()
        if not row:
            conn.close()
            return jsonify({"matches": [], "note": "No odds snapshots yet. Go to Matchup Report and Fetch odds first."})
        ts = row["ts"]
        rows = conn.execute(
            "SELECT payload_json FROM odds_snapshots WHERE ts = ? LIMIT 2000",
            (ts,),
        ).fetchall()
        conn.close()

        matches = []
        for r in rows:
            m = json.loads(r["payload_json"])
            mid = normalize_match(m)
            bm = (m.get("bookmakers") or [])
            first = bm[0] if bm else None
            h2h = None
            if first:
                for market in first.get("markets") or []:
                    if market.get("key") == "h2h":
                        h2h = market
                        break
            # grab moneyline prices for both outcomes (decimal)
            outcomes = []
            if h2h:
                for o in h2h.get("outcomes") or []:
                    outcomes.append({"name": o.get("name"), "odds_decimal": o.get("price")})
            matches.append({
                "match_id": mid,
                "sport_key": m.get("sport_key"),
                "tournament": m.get("sport_title"),
                "commence_time": m.get("commence_time"),
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "bookmaker": first.get("title") if first else None,
                "outcomes": outcomes,
            })

        return jsonify({"ts": ts, "matches": matches})

    @app.get("/api/candidates/latest")
    def api_candidates_latest():
        """Fetch ranked candidates from DB without calling Odds API.

        Params:
          snapshot_id: optional snapshot id (defaults to latest)
          view: actionable|watchlist|debug|all (default all)
          limit: int (default 50)
        """
        view = (request.args.get("view") or "all").lower()
        limit = int(request.args.get("limit") or 50)
        limit = max(1, min(500, limit))
        snapshot_id = request.args.get("snapshot_id")

        conn = connect()
        if snapshot_id:
            snap = conn.execute(
                "SELECT id, ts, sport_key, markets, regions, top_n, refresh_interval_sec, model_version FROM snapshots WHERE id = ?",
                (int(snapshot_id),),
            ).fetchone()
        else:
            snap = conn.execute(
                "SELECT id, ts, sport_key, markets, regions, top_n, refresh_interval_sec, model_version FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not snap:
            conn.close()
            return jsonify({"snapshot": None, "candidates": [], "note": "No snapshots yet. Call /api/odds first."})

        where = ""
        if view == "actionable":
            where = " AND actionable = 1"
        elif view == "watchlist":
            where = " AND view_mode = 'watchlist'"
        elif view == "debug":
            where = " AND view_mode = 'debug'"
        elif view == "all":
            where = ""
        else:
            conn.close()
            return jsonify({"error": "invalid view (use actionable|watchlist|debug|all)"}), 400

        rows = conn.execute(
            """
            SELECT *
            FROM ranked_candidates
            WHERE snapshot_id = ?
            """ + where + "\nORDER BY ev_adj DESC NULLS LAST, confidence DESC NULLS LAST\nLIMIT ?",
            (snap["id"], limit),
        ).fetchall()
        conn.close()

        candidates = []
        for r in rows:
            d = dict(r)
            if d.get("matchup_flags_json"):
                try:
                    payload = json.loads(d.get("matchup_flags_json") or "{}")
                    d["component_scores"] = payload.get("component_scores") or {}
                    d["axis_notes"] = payload.get("axis_notes") or {}
                    if not d.get("reasons"):
                        d["reasons"] = payload.get("reasons") or []
                except Exception:
                    d["component_scores"] = {}
                    d["axis_notes"] = {}
            candidates.append(d)

        return jsonify({
            "snapshot": dict(snap),
            "count": len(rows),
            "candidates": candidates,
        })

    @app.get("/api/matchups/daily")
    def api_matchups_daily():
        date_et = request.args.get("date") or _current_et_date()
        auto_fetch = (request.args.get("auto_fetch") or "1") == "1"
        conn = connect()
        snap = _latest_daily_snapshot(conn, date_et)
        conn.close()

        if not snap and auto_fetch:
            try:
                sports, _headers = list_sports()
                atp = [s for s in (sports or []) if s.get("active") and str(s.get("key") or "").startswith("tennis_atp")]
                atp = sorted(atp, key=lambda s: str(s.get("key") or ""))
                if not atp:
                    return jsonify({"error": "no_active_atp_feed", "date_et": date_et}), 404
                sport_key = atp[0]["key"]
                with app.test_request_context(f"/api/odds?sport_key={sport_key}&markets=h2h&capture_daily=1"):
                    resp = api_odds()
                if isinstance(resp, tuple):
                    response, status = resp
                    return response, status
                data = resp.get_json()
                summary = _match_status_counts(data.get("outputs") or [])
                summary["date_et"] = date_et
                conn = connect()
                summary["actionables"] = _daily_actionable_stats(conn, date_et)
                conn.close()
                data["date_et"] = date_et
                data["slate_summary"] = summary
                return jsonify(data)
            except Exception as e:
                return jsonify({"error": "daily_fetch_failed", "message": str(e), "date_et": date_et}), 500

        if not snap:
            return jsonify({"date_et": date_et, "snapshot": None, "candidates_debug": [], "candidates_watchlist": [], "candidates_actionable": []})

        conn = connect()
        rows_all = conn.execute(
            "SELECT * FROM ranked_candidates WHERE snapshot_id = ? ORDER BY ev_adj DESC NULLS LAST, confidence DESC NULLS LAST LIMIT 300",
            (snap["id"],),
        ).fetchall()
        payload_rows = conn.execute(
            "SELECT match_id, payload_json FROM odds_snapshots WHERE ts = ? AND sport_key = ?",
            (snap["ts"], snap["sport_key"]),
        ).fetchall()
        actionables_stats = _daily_actionable_stats(conn, date_et)
        watchlist_stats = _daily_watchlist_stats(conn, date_et)
        conn.close()

        thresholds = _selection_thresholds()

        def row_to_dict(r):
            d = dict(r)
            component_scores = {}
            axis_notes = {}
            if d.get("matchup_flags_json"):
                try:
                    payload = json.loads(d.get("matchup_flags_json") or "{}")
                    component_scores = payload.get("component_scores") or {}
                    axis_notes = payload.get("axis_notes") or {}
                except Exception:
                    component_scores = {}
                    axis_notes = {}
            d["component_scores"] = component_scores
            d["axis_notes"] = axis_notes
            item = SimpleNamespace(**d)
            live_view_mode, is_actionable = _classify_candidate(item, thresholds)
            tier, units = _assign_tier_and_units(item, "actionable" if is_actionable else live_view_mode, thresholds)
            strategy_meta = _build_strategy_reasoning(item, "actionable" if is_actionable else live_view_mode, thresholds)
            d["actionable"] = 1 if is_actionable else 0
            d["view_mode"] = live_view_mode
            d["selection_tier"] = tier
            d["units_suggested"] = units if is_actionable else None
            d["strategy_summary"] = strategy_meta.get("summary")
            d["strategy_why"] = strategy_meta.get("why") or []
            d["strategy_risk"] = strategy_meta.get("risk") or []
            d["alternative_market"] = strategy_meta.get("alternative_market") or None
            d["reasons"] = d.get("reasons") or []
            return d

        payloads = []
        for r in payload_rows:
            try:
                m = json.loads(r["payload_json"])
                payloads.append({
                    "match_id": r["match_id"],
                    "commence_time": m.get("commence_time"),
                    "completed": False,
                })
            except Exception:
                pass

        all_dicts = [row_to_dict(r) for r in rows_all]
        rows_actionable = [r for r in all_dicts if r.get("actionable") == 1][:100]
        rows_watchlist = [r for r in all_dicts if r.get("view_mode") == "watchlist"][:10]

        summary = _match_status_counts(payloads)
        summary["date_et"] = date_et
        summary["actionables"] = actionables_stats
        summary["watchlist"] = watchlist_stats

        return jsonify({
            "date_et": date_et,
            "snapshot": dict(snap),
            "slate_summary": summary,
            "candidates_debug": all_dicts,
            "candidates_watchlist": rows_watchlist,
            "candidates_actionable": rows_actionable,
        })

    @app.get("/api/snapshots/recent")
    def api_snapshots_recent():
        """Return recent refresh snapshots with quick candidate counts for UI browsing."""
        limit = int(request.args.get("limit") or 12)
        limit = max(1, min(100, limit))

        conn = connect()
        rows = conn.execute(
            """
            SELECT
              s.id,
              s.ts,
              s.sport_key,
              s.markets,
              s.regions,
              s.top_n,
              s.refresh_interval_sec,
              s.model_version,
              COUNT(rc.id) AS candidate_count,
              SUM(CASE WHEN rc.actionable = 1 THEN 1 ELSE 0 END) AS actionable_count,
              MAX(rc.ev_adj) AS best_ev_adj,
              MAX(rc.confidence) AS best_confidence
            FROM snapshots s
            LEFT JOIN ranked_candidates rc ON rc.snapshot_id = s.id
            GROUP BY s.id, s.ts, s.sport_key, s.markets, s.regions, s.top_n, s.refresh_interval_sec, s.model_version
            ORDER BY s.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return jsonify({"count": len(rows), "snapshots": [dict(r) for r in rows]})

    @app.get("/api/candidate/clv")
    def api_candidate_clv():
        """Return CLV history for a stored candidate id."""
        candidate_id = request.args.get("candidate_id")
        if not candidate_id:
            return jsonify({"error": "candidate_id required"}), 400

        conn = connect()
        cand = conn.execute(
            """
            SELECT id, snapshot_id, match_id, commence_time, player_a, player_b, side, market_type,
                   line, price_decimal, price_american, book, confidence, ev, ev_adj
            FROM ranked_candidates
            WHERE id = ?
            """,
            (int(candidate_id),),
        ).fetchone()
        if not cand:
            conn.close()
            return jsonify({"error": "candidate not found"}), 404

        rows = conn.execute(
            """
            SELECT ts, minutes_before_start, best_price_decimal, consensus_price_decimal, best_line, consensus_line
            FROM clv_snapshots
            WHERE candidate_id = ?
            ORDER BY ts ASC
            """,
            (int(candidate_id),),
        ).fetchall()
        conn.close()

        return jsonify({
            "candidate": dict(cand),
            "count": len(rows),
            "rows": [dict(r) for r in rows],
        })

    @app.get("/api/actionables")
    def api_actionables():
        """Return ET-day captured actionables based on commence_time date in ET."""
        date_et = (request.args.get("date") or "").strip() or _current_et_date()
        conn = connect()
        _settle_open_actionables(conn, days_from=1)
        conn.commit()

        rows = conn.execute(
            "SELECT * FROM daily_actionables ORDER BY ev_adj DESC NULLS LAST, id DESC LIMIT 500"
        ).fetchall()
        conn.close()

        now_utc = datetime.now(timezone.utc)
        out = []
        for r in rows:
            d = dict(r)
            if _et_date_from_iso(d.get("commence_time")) != date_et:
                continue
            status_display = d.get("result") or "OPEN"
            status_note = None
            if status_display == "OPEN":
                commence_time = d.get("commence_time")
                if commence_time:
                    try:
                        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                        if dt <= now_utc:
                            status_display = "CLOSED"
                            status_note = "Start time has passed; awaiting score-based grading."
                    except Exception:
                        pass
            d["status_display"] = status_display
            d["status_note"] = status_note
            out.append(d)

        return jsonify({"date_et": date_et, "count": len(out), "rows": out})

    @app.get("/api/strategy/audit")
    def api_strategy_audit():
        conn = connect()
        audit = _strategy_audit_summary(conn)
        conn.close()
        return jsonify(audit)

    @app.get("/api/style/player")
    def api_style_player():
        """Lookup a player's MCP style profile (men).

        Params:
          q: player name (substring)
          exact: 1 to require exact match
        """
        q = (request.args.get("q") or "").strip()
        exact = (request.args.get("exact") or "0") == "1"
        if not q:
            return jsonify({"error": "q required"}), 400

        conn = connect()
        if exact:
            row = conn.execute("SELECT * FROM style_mcp_player_m WHERE player = ?", (q,)).fetchone()
            conn.close()
            return jsonify({"match": dict(row) if row else None})

        rows = conn.execute(
            "SELECT * FROM style_mcp_player_m WHERE player LIKE ? ORDER BY matches DESC LIMIT 20",
            (f"%{q}%",),
        ).fetchall()
        conn.close()
        return jsonify({"count": len(rows), "matches": [dict(r) for r in rows]})

    def infer_surface(sport_key: str) -> str | None:
        # v1 heuristic; we can upgrade later with tournament metadata.
        if "indian_wells" in (sport_key or ""):
            return "Hard"
        if "wimbledon" in (sport_key or ""):
            return "Grass"
        if "roland" in (sport_key or "") or "french_open" in (sport_key or ""):
            return "Clay"
        return None

    def _current_et_date() -> str:
        return datetime.now(timezone.utc).astimezone(ET).date().isoformat()

    def _et_date_from_iso(ts: str | None) -> str | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET).date().isoformat()
        except Exception:
            return None

    def _daily_result_stats(conn, table_name: str, date_et: str):
        rows = conn.execute(
            f"SELECT result, commence_time FROM {table_name} ORDER BY id DESC LIMIT 500"
        ).fetchall()
        total = wins = losses = open_n = 0
        for row in rows:
            if _et_date_from_iso(row["commence_time"]) != date_et:
                continue
            total += 1
            result = row["result"] or "OPEN"
            if result == "WIN":
                wins += 1
            elif result == "LOSS":
                losses += 1
            else:
                open_n += 1
        denom = wins + losses
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "open_n": open_n,
            "win_rate": round(wins / denom, 3) if denom > 0 else None,
        }

    def _daily_actionable_stats(conn, date_et: str):
        return _daily_result_stats(conn, "daily_actionables", date_et)

    def _daily_watchlist_stats(conn, date_et: str):
        return _daily_result_stats(conn, "daily_watchlist", date_et)

    def _match_status_counts(payload_rows: list[dict]):
        now_utc = datetime.now(timezone.utc)
        out = {"yet_to_start": 0, "live": 0, "ended": 0, "total_matches": 0}
        seen = set()
        for row in payload_rows:
            match_id = row.get("match_id")
            if match_id in seen:
                continue
            seen.add(match_id)
            out["total_matches"] += 1
            commence_time = row.get("commence_time")
            completed = row.get("completed")
            if completed:
                out["ended"] += 1
                continue
            try:
                dt = datetime.fromisoformat((commence_time or "").replace("Z", "+00:00"))
            except Exception:
                out["yet_to_start"] += 1
                continue
            if dt > now_utc:
                out["yet_to_start"] += 1
            elif dt + timedelta(minutes=MATCH_LIVE_WINDOW_MIN) > now_utc:
                out["live"] += 1
            else:
                out["ended"] += 1
        return out

    def _latest_daily_snapshot(conn, date_et: str):
        rows = conn.execute(
            "SELECT id, ts, sport_key, markets, regions, top_n, refresh_interval_sec, model_version FROM snapshots WHERE sport_key LIKE 'tennis_atp%' ORDER BY id DESC LIMIT 200"
        ).fetchall()
        for row in rows:
            if _et_date_from_iso(row["ts"]) == date_et:
                return row
        return None

    def _selection_thresholds():
        return {
            "actionable_max_odds": float(os.getenv("ACTIONABLE_MAX_ODDS_DEC") or 5.0),
            "actionable_min_conf": float(os.getenv("ACTIONABLE_MIN_CONF") or 0.66),
            "actionable_min_ev": float(os.getenv("ACTIONABLE_MIN_EV") or 0.03),
            "actionable_min_ev_adj": float(os.getenv("ACTIONABLE_MIN_EV_ADJ") or 0.03),
            "actionable_max_edge": float(os.getenv("ACTIONABLE_MAX_EDGE") or 0.10),
            "actionable_min_matchup": float(os.getenv("ACTIONABLE_MIN_MATCHUP") or 60.0),
            "actionable_min_market_value": float(os.getenv("ACTIONABLE_MIN_MARKET_VALUE") or 54.0),
            "actionable_min_reliability": float(os.getenv("ACTIONABLE_MIN_RELIABILITY") or 58.0),
            "watchlist_max_odds": float(os.getenv("WATCHLIST_MAX_ODDS_DEC") or 6.0),
            "watchlist_min_conf": float(os.getenv("WATCHLIST_MIN_CONF") or 0.60),
            "watchlist_min_ev": float(os.getenv("WATCHLIST_MIN_EV") or 0.02),
            "watchlist_min_ev_adj": float(os.getenv("WATCHLIST_MIN_EV_ADJ") or 0.02),
            "watchlist_max_edge": float(os.getenv("WATCHLIST_MAX_EDGE") or 0.10),
            "watchlist_min_matchup": float(os.getenv("WATCHLIST_MIN_MATCHUP") or 54.0),
            "watchlist_min_market_value": float(os.getenv("WATCHLIST_MIN_MARKET_VALUE") or 48.0),
            "watchlist_min_reliability": float(os.getenv("WATCHLIST_MIN_RELIABILITY") or 52.0),
            "heavy_favorite_max_price": float(os.getenv("HEAVY_FAVORITE_MAX_PRICE_DEC") or 1.20),
            "heavy_favorite_min_conf": float(os.getenv("HEAVY_FAVORITE_MIN_CONF") or 0.74),
            "heavy_favorite_min_ev": float(os.getenv("HEAVY_FAVORITE_MIN_EV") or 0.08),
            "heavy_favorite_min_ev_adj": float(os.getenv("HEAVY_FAVORITE_MIN_EV_ADJ") or 0.055),
            "tie_ev_band": float(os.getenv("SELECTION_TIE_EV_BAND") or 0.015),
            "tie_conf_band": float(os.getenv("SELECTION_TIE_CONF_BAND") or 0.04),
            "tier_a_ev_adj": float(os.getenv("TIER_A_MIN_EV_ADJ") or 0.09),
            "tier_a_conf": float(os.getenv("TIER_A_MIN_CONF") or 0.74),
            "tier_b_ev_adj": float(os.getenv("TIER_B_MIN_EV_ADJ") or 0.06),
            "tier_b_conf": float(os.getenv("TIER_B_MIN_CONF") or 0.70),
            "tier_c_ev_adj": float(os.getenv("TIER_C_MIN_EV_ADJ") or 0.04),
            "tier_c_conf": float(os.getenv("TIER_C_MIN_CONF") or 0.66),
            "tier_d_ev_adj": float(os.getenv("TIER_D_MIN_EV_ADJ") or 0.03),
            "tier_d_conf": float(os.getenv("TIER_D_MIN_CONF") or 0.64),
        }

    def _is_heavy_favorite_ml(c, thresholds: dict) -> bool:
        return (
            (getattr(c, "market_type", None) == "ML")
            and (getattr(c, "price_decimal", None) is not None)
            and float(c.price_decimal) <= thresholds["heavy_favorite_max_price"]
        )

    def _selection_sort_key(c, thresholds: dict):
        ev_adj = float(getattr(c, "ev_adj", -999) or -999)
        conf = float(getattr(c, "confidence", 0.0) or 0.0)
        price = float(getattr(c, "price_decimal", 999.0) or 999.0)
        heavy_fav = 1 if _is_heavy_favorite_ml(c, thresholds) else 0
        ev_bucket = round(ev_adj / max(thresholds["tie_ev_band"], 0.001))
        conf_bucket = round(conf / max(thresholds["tie_conf_band"], 0.01))
        return (-ev_bucket, -conf_bucket, heavy_fav, price, -ev_adj, -conf)

    def _assign_tier_and_units(c, view_mode: str, thresholds: dict) -> tuple[str, float | None]:
        conf = float(getattr(c, "confidence", 0.0) or 0.0)
        ev_adj = float(getattr(c, "ev_adj", 0.0) or 0.0)
        if view_mode != "actionable":
            return ("Watchlist", None) if view_mode == "watchlist" else ("Debug", None)
        if conf >= thresholds["tier_a_conf"] and ev_adj >= thresholds["tier_a_ev_adj"]:
            return "Tier A", 4.0
        if conf >= thresholds["tier_b_conf"] and ev_adj >= thresholds["tier_b_ev_adj"]:
            return "Tier B", 3.0
        if conf >= thresholds["tier_c_conf"] and ev_adj >= thresholds["tier_c_ev_adj"]:
            return "Tier C", 2.0
        return "Tier D", 1.0

    def _best_alternative_market(match_payload: dict, side: str):
        best = None
        best_score = None
        for bm in (match_payload.get("bookmakers") or []):
            for market in (bm.get("markets") or []):
                key = market.get("key")
                if key not in ("spreads", "totals"):
                    continue
                for o in (market.get("outcomes") or []):
                    if key == "spreads" and o.get("name") != side:
                        continue
                    price = o.get("price")
                    if not price:
                        continue
                    american = dec_to_american(price)
                    score = abs((american or 0) - 100)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = {
                            "market": key,
                            "name": o.get("name"),
                            "line": o.get("point"),
                            "price_decimal": price,
                            "price_american": american,
                            "book": bm.get("title"),
                        }
        return best

    def _build_strategy_reasoning(c, view_mode: str, thresholds: dict, alt_market=None):
        price_am = dec_to_american(getattr(c, "price_decimal", None))
        tier, units = _assign_tier_and_units(c, view_mode, thresholds)
        model_edge = ((getattr(c, "p_final", 0.0) or 0.0) - (getattr(c, "q_implied", 0.0) or 0.0))
        matchup_strength = float(getattr(c, "matchup_strength", 0.0) or 0.0)
        market_value = float(getattr(c, "market_value", 0.0) or 0.0)
        reliability = float(getattr(c, "reliability", 0.0) or 0.0)
        component_scores = getattr(c, "component_scores", None) or {}

        summary = (
            f"{c.side} grades as Matchup {matchup_strength:.0f}/100, Market {market_value:.0f}/100, "
            f"Reliability {reliability:.0f}/100. Model makes {((c.p_final or 0)*100):.1f}% vs market {((c.q_implied or 0)*100):.1f}% "
            f"at {price_am if price_am is not None else '—'} — edge {model_edge*100:+.1f} pts."
        )

        why = []
        risk = []

        if view_mode == "actionable":
            why.append(f"{tier} actionable — {units:.1f}u suggested because all three axes cleared the live thresholds.")
        elif view_mode == "watchlist":
            why.append("Watchlist only — some edge exists, but one or more axes are not strong enough for full actionable status.")
        else:
            why.append("Debug only — current framework does not trust the full matchup/market/reliability stack enough yet.")

        why.append(
            f"Matchup stack: player quality {component_scores.get('player_quality', '—')}, surface {component_scores.get('surface_strength', '—')}, "
            f"recent form {component_scores.get('recent_form', '—')}, serve/return {component_scores.get('serve_return_profile', '—')}, style {component_scores.get('style_interaction', '—')}."
        )
        why.append(
            f"Market stack: movement {component_scores.get('open_close_comparison', '—')}, fair-vs-implied {component_scores.get('implied_vs_fair_probability', '—')}, price bucket {component_scores.get('price_bucket_viability', '—')}."
        )
        why.append(
            f"Reliability stack: sample {component_scores.get('sample_size', '—')}, calibration {component_scores.get('calibration', '—')}, data completeness {component_scores.get('data_completeness', '—')}."
        )

        if _is_heavy_favorite_ml(c, thresholds):
            risk.append("Heavy-favorite ML tax — expensive moneyline needs stronger matchup and reliability support than a normal ML.")
            if alt_market:
                why.append(f"Alternative market to consider: {alt_market['market'].upper()} {alt_market.get('line')} on {alt_market.get('name')} at {alt_market.get('price_american')} ({alt_market.get('book')}).")
            else:
                why.append("Alternative market check: prefer same-player spread / alt line if available instead of laying a heavy ML.")
        elif (c.price_decimal or 99) >= 3.5:
            risk.append("Dog/longer price — edge is more fragile and should be sized carefully.")

        if (c.price_decimal or 99) > thresholds["watchlist_max_odds"]:
            risk.append("Longshot filter — current strategy no longer wants very high-priced dogs on the watchlist.")
        if (c.confidence or 0) < 0.70:
            risk.append("Confidence is decent but not elite — do not treat this like a top-tier play.")
        if matchup_strength < thresholds["actionable_min_matchup"]:
            risk.append("Matchup case is not strong enough yet — the tennis read is weaker than we want.")
        if market_value < thresholds["actionable_min_market_value"]:
            risk.append("Price case is thin — the market is not clearly underpricing the matchup enough.")
        if reliability < thresholds["actionable_min_reliability"]:
            risk.append("Reliability is middling — sample / calibration / completeness still need more trust.")
        if abs(model_edge) >= thresholds["actionable_max_edge"] * 0.8:
            risk.append("Model is disagreeing with market materially — upside is real, but calibration risk is higher.")

        return {
            "tier": tier,
            "units": units,
            "summary": summary,
            "why": why,
            "risk": risk,
            "alternative_market": alt_market,
        }

    def _classify_candidate(c, thresholds: dict) -> tuple[str, bool]:
        if c.price_decimal is None or c.ev is None or c.ev_adj is None or c.q_implied is None or c.p_final is None:
            return "debug", False

        model_edge = abs((c.p_final or 0.0) - (c.q_implied or 0.0))
        matchup_strength = float(getattr(c, "matchup_strength", 0.0) or 0.0)
        market_value = float(getattr(c, "market_value", 0.0) or 0.0)
        reliability = float(getattr(c, "reliability", 0.0) or 0.0)

        actionable = (
            c.price_decimal <= thresholds["actionable_max_odds"]
            and (c.confidence or 0.0) >= thresholds["actionable_min_conf"]
            and c.ev >= thresholds["actionable_min_ev"]
            and c.ev_adj >= thresholds["actionable_min_ev_adj"]
            and model_edge <= thresholds["actionable_max_edge"]
            and matchup_strength >= thresholds["actionable_min_matchup"]
            and market_value >= thresholds["actionable_min_market_value"]
            and reliability >= thresholds["actionable_min_reliability"]
        )
        if actionable:
            if _is_heavy_favorite_ml(c, thresholds):
                heavy_ok = (
                    (c.confidence or 0.0) >= thresholds["heavy_favorite_min_conf"]
                    and c.ev >= thresholds["heavy_favorite_min_ev"]
                    and c.ev_adj >= thresholds["heavy_favorite_min_ev_adj"]
                    and matchup_strength >= thresholds["actionable_min_matchup"] + 4.0
                    and reliability >= thresholds["actionable_min_reliability"] + 4.0
                )
                if heavy_ok:
                    return "actionable", True
                return "watchlist", False
            return "actionable", True

        watchlist = (
            c.price_decimal <= thresholds["watchlist_max_odds"]
            and (c.confidence or 0.0) >= thresholds["watchlist_min_conf"]
            and c.ev >= thresholds["watchlist_min_ev"]
            and c.ev_adj >= thresholds["watchlist_min_ev_adj"]
            and model_edge <= thresholds["watchlist_max_edge"]
            and matchup_strength >= thresholds["watchlist_min_matchup"]
            and market_value >= thresholds["watchlist_min_market_value"]
            and reliability >= thresholds["watchlist_min_reliability"]
        )
        if watchlist:
            return "watchlist", False

        return "debug", False

    def _strategy_audit_summary(conn):
        latest = conn.execute(
            "SELECT id FROM snapshots ORDER BY id DESC LIMIT 50"
        ).fetchall()
        latest_ids = [int(r[0]) for r in latest]

        summary = {
            "latest_snapshot_count": len(latest_ids),
            "latest_actionables": {},
            "latest_watchlist": {},
            "settled_actionables": {},
            "notes": [],
        }
        thresholds = _selection_thresholds()
        summary["thresholds"] = thresholds
        summary["recent_changes"] = [
            "Tier-based unit framework is active (Tier A-D + Watchlist/Debug).",
            "Actionable cards now show strategy summary, why text, and main risk notes.",
            "Heavy-favorite ML logic now suggests same-player alternative markets when available.",
            "Price-aware ranking now favors cheaper bets when EV/confidence are in the same band.",
            "Bet Tracker and Matchup Report both have live-refresh logic, but settled sample is still small.",
        ]
        if not latest_ids:
            summary["notes"].append("No snapshots available yet.")
            return summary

        placeholders = ",".join("?" for _ in latest_ids)
        rows = conn.execute(
            f"""
            SELECT id, snapshot_id, market_type, price_decimal, price_american, confidence, ev, ev_adj, p_final, q_implied,
                   matchup_strength, market_value, reliability
            FROM ranked_candidates
            WHERE snapshot_id IN ({placeholders})
            """,
            tuple(latest_ids),
        ).fetchall()

        actionables = []
        watchlist = []
        for row in rows:
            item = SimpleNamespace(**dict(row))
            view_mode, is_actionable = _classify_candidate(item, thresholds)
            tier, units = _assign_tier_and_units(item, "actionable" if is_actionable else view_mode, thresholds)
            out = dict(row)
            out["view_mode_live"] = "actionable" if is_actionable else view_mode
            out["selection_tier"] = tier
            out["units_suggested_live"] = units
            if is_actionable:
                actionables.append(out)
            elif view_mode == "watchlist":
                watchlist.append(out)

        def summarize(items):
            if not items:
                return {
                    "n": 0,
                    "avg_odds": None,
                    "max_odds": None,
                    "avg_conf": None,
                    "avg_ev": None,
                    "avg_ev_adj": None,
                    "avg_matchup_strength": None,
                    "avg_market_value": None,
                    "avg_reliability": None,
                }
            odds = [x["price_decimal"] for x in items if x.get("price_decimal") is not None]
            confs = [x["confidence"] for x in items if x.get("confidence") is not None]
            evs = [x["ev"] for x in items if x.get("ev") is not None]
            ev_adjs = [x["ev_adj"] for x in items if x.get("ev_adj") is not None]
            matchup = [x["matchup_strength"] for x in items if x.get("matchup_strength") is not None]
            market_vals = [x["market_value"] for x in items if x.get("market_value") is not None]
            reliab = [x["reliability"] for x in items if x.get("reliability") is not None]
            return {
                "n": len(items),
                "avg_odds": round(sum(odds) / len(odds), 2) if odds else None,
                "max_odds": round(max(odds), 2) if odds else None,
                "avg_conf": round(sum(confs) / len(confs), 3) if confs else None,
                "avg_ev": round(sum(evs) / len(evs), 3) if evs else None,
                "avg_ev_adj": round(sum(ev_adjs) / len(ev_adjs), 3) if ev_adjs else None,
                "avg_matchup_strength": round(sum(matchup) / len(matchup), 1) if matchup else None,
                "avg_market_value": round(sum(market_vals) / len(market_vals), 1) if market_vals else None,
                "avg_reliability": round(sum(reliab) / len(reliab), 1) if reliab else None,
            }

        def safe_win_rate(wins, losses):
            denom = (wins or 0) + (losses or 0)
            return round((wins or 0) / denom, 3) if denom > 0 else None

        def price_bucket_from_decimal(price):
            if price is None:
                return "unknown"
            if price < 1.8:
                return "favorite_lt_1.8"
            if price < 2.5:
                return "short_1.8_2.5"
            return "dog_2.5_plus"

        summary["latest_actionables"] = summarize(actionables)
        summary["latest_watchlist"] = summarize(watchlist)

        buckets = {}
        for item in actionables:
            bucket = price_bucket_from_decimal(item.get("price_decimal"))
            buckets.setdefault(bucket, []).append(item)
        summary["latest_actionables"]["odds_buckets"] = [
            {
                "bucket": bucket,
                "n": len(items),
                "avg_ev_adj": round(sum(x["ev_adj"] for x in items if x.get("ev_adj") is not None) / len(items), 3),
                "avg_conf": round(sum(x["confidence"] for x in items if x.get("confidence") is not None) / len(items), 3),
            }
            for bucket, items in sorted(buckets.items())
        ]

        tier_counts = {}
        for item in actionables:
            tier_counts[item["selection_tier"]] = tier_counts.get(item["selection_tier"], 0) + 1
        summary["tier_mix"] = [{"tier": k, "count": v} for k, v in sorted(tier_counts.items())]

        settled = conn.execute(
            """
            SELECT
              COUNT(*) AS settled_n,
              SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) AS losses,
              ROUND(AVG(price_decimal), 2) AS avg_odds,
              ROUND(AVG(confidence), 3) AS avg_conf,
              ROUND(AVG(ev_adj), 3) AS avg_ev_adj
            FROM daily_actionables
            WHERE result IN ('WIN', 'LOSS')
            """
        ).fetchone()
        summary["settled_actionables"] = dict(settled) if settled else {}

        settled_n = int((settled or {"settled_n": 0})["settled_n"] or 0)
        if settled_n > 0:
            wins = int((settled or {"wins": 0})["wins"] or 0)
            losses = int((settled or {"losses": 0})["losses"] or 0)
            summary["settled_actionables"]["win_rate"] = safe_win_rate(wins, losses)
        else:
            summary["notes"].append("Not enough settled actionables yet for a real threshold backtest.")

        settled_join = conn.execute(
            """
            SELECT da.result, da.market_type, da.price_decimal, da.confidence, da.ev_adj, rc.units_suggested
            FROM daily_actionables da
            LEFT JOIN ranked_candidates rc ON rc.id = da.candidate_id
            WHERE da.result IN ('WIN', 'LOSS')
            """
        ).fetchall()

        tier_perf = {}
        market_perf = {}
        price_perf = {}
        conf_perf = {}
        sizing_rows = []

        for row in settled_join:
            row = dict(row)
            candidate_like = SimpleNamespace(
                confidence=row.get("confidence"),
                ev_adj=row.get("ev_adj"),
                price_decimal=row.get("price_decimal"),
            )
            tier, units = _assign_tier_and_units(candidate_like, "actionable", thresholds)
            res = row.get("result")
            win = 1 if res == "WIN" else 0
            loss = 1 if res == "LOSS" else 0

            t = tier_perf.setdefault(tier, {"tier": tier, "bets": 0, "wins": 0, "losses": 0, "win_rate": None, "pnl_units": 0.0, "avg_units": 0.0, "_units": []})
            t["bets"] += 1; t["wins"] += win; t["losses"] += loss
            eff_units = float(row.get("units_suggested") or units or 0.0)
            pnl_units = eff_units if res == "WIN" else (-eff_units if res == "LOSS" else 0.0)
            t["pnl_units"] += pnl_units; t["_units"].append(eff_units)

            mkt = row.get("market_type") or "unknown"
            m = market_perf.setdefault(mkt, {"market": mkt, "bets": 0, "wins": 0, "losses": 0, "win_rate": None})
            m["bets"] += 1; m["wins"] += win; m["losses"] += loss

            pb = price_bucket_from_decimal(row.get("price_decimal"))
            p = price_perf.setdefault(pb, {"bucket": pb, "bets": 0, "wins": 0, "losses": 0, "win_rate": None})
            p["bets"] += 1; p["wins"] += win; p["losses"] += loss

            conf = row.get("confidence")
            if conf is None:
                cb = "unknown"
            elif conf < 0.65:
                cb = "conf_lt_65"
            elif conf < 0.72:
                cb = "conf_65_72"
            else:
                cb = "conf_72_plus"
            c = conf_perf.setdefault(cb, {"bucket": cb, "bets": 0, "wins": 0, "losses": 0, "win_rate": None})
            c["bets"] += 1; c["wins"] += win; c["losses"] += loss

            sizing_rows.append({"tier": tier, "units": eff_units, "result": res})

        for coll in (tier_perf, market_perf, price_perf, conf_perf):
            for val in coll.values():
                val["win_rate"] = safe_win_rate(val.get("wins", 0), val.get("losses", 0))
                if "_units" in val:
                    units_list = val.pop("_units")
                    val["avg_units"] = round(sum(units_list) / len(units_list), 2) if units_list else 0.0
                    val["pnl_units"] = round(val["pnl_units"], 2)

        summary["tier_performance"] = list(sorted(tier_perf.values(), key=lambda x: x["tier"]))
        summary["market_type_performance"] = list(sorted(market_perf.values(), key=lambda x: x["market"]))
        summary["favorites_vs_dogs"] = list(sorted(price_perf.values(), key=lambda x: x["bucket"]))
        summary["confidence_bucket_performance"] = list(sorted(conf_perf.values(), key=lambda x: x["bucket"]))

        sizing_notes = []
        if sizing_rows:
            high_unit_rows = [r for r in sizing_rows if (r["units"] or 0) >= 3]
            if high_unit_rows:
                wins = sum(1 for r in high_unit_rows if r["result"] == "WIN")
                losses = sum(1 for r in high_unit_rows if r["result"] == "LOSS")
                sizing_notes.append(f"Higher-size bets (3u+) currently have win rate {((wins/(wins+losses))*100):.1f}% across {wins+losses} settled bets." if (wins+losses) else "Higher-size bets have no settled sample yet.")
        else:
            sizing_notes.append("Not enough settled bets to know whether suggested units are too aggressive or too small yet.")
        summary["sizing_feedback"] = sizing_notes

        clv_rows = conn.execute(
            """
            SELECT rc.units_suggested, cs.best_price_decimal, cs.consensus_price_decimal
            FROM clv_snapshots cs
            JOIN ranked_candidates rc ON rc.id = cs.candidate_id
            ORDER BY cs.id DESC
            LIMIT 500
            """
        ).fetchall()
        clv_by_tier = {}
        for row in clv_rows:
            row = dict(row)
            candidate_like = SimpleNamespace(confidence=0.0, ev_adj=0.0, price_decimal=None)
            units = float(row.get("units_suggested") or 0.0)
            tier = "Tier A" if units >= 4 else ("Tier B" if units >= 3 else ("Tier C" if units >= 2 else "Tier D"))
            if row.get("best_price_decimal") and row.get("consensus_price_decimal"):
                clv = float(row["best_price_decimal"]) - float(row["consensus_price_decimal"])
                rec = clv_by_tier.setdefault(tier, {"tier": tier, "rows": 0, "avg_clv_dec": 0.0})
                rec["rows"] += 1
                rec["avg_clv_dec"] += clv
        summary["clv_by_tier"] = []
        for tier, rec in sorted(clv_by_tier.items()):
            if rec["rows"]:
                rec["avg_clv_dec"] = round(rec["avg_clv_dec"] / rec["rows"], 4)
            summary["clv_by_tier"].append(rec)

        summary["data_tracking"] = {
            "settled_actionables": settled_n,
            "paper_bets": int((conn.execute("SELECT COUNT(*) AS n FROM paper_bets").fetchone() or {"n": 0})["n"] or 0),
            "paper_bets_open": int((conn.execute("SELECT COUNT(*) AS n FROM paper_bets WHERE COALESCE(result, 'OPEN') = 'OPEN'").fetchone() or {"n": 0})["n"] or 0),
            "clv_rows": int((conn.execute("SELECT COUNT(*) AS n FROM clv_snapshots").fetchone() or {"n": 0})["n"] or 0),
        }

        summary["strategy_framework"] = [
            {"market": "ML", "status": "active", "notes": "Primary live strategy. Tiered units, heavy-favorite guardrails, watchlist split, price-aware ranking are active."},
            {"market": "spreads", "status": "next", "notes": "Needed for heavy-favorite substitution and better favorite expression; selection/grading rules still need dedicated modeling."},
            {"market": "totals", "status": "next", "notes": "Should get separate thresholds and grading rules; not reliable enough to surface as true strategy output yet."},
        ]
        summary["notes"].append("Audit applies the current selection thresholds to the most recent 50 snapshots, even if those snapshots were generated before the new rules shipped.")
        summary["notes"].append("Current actionable design favors lower-variance selections; high-EV fragile dogs are demoted to watchlist.")
        summary["notes"].append("Tier sizing is now built in: stronger EV_adj/confidence combinations map to larger suggested units, while watchlist/debug carry no suggested size.")
        return summary

    def player_id_from_name(conn, name: str | None):
        if not name:
            return None
        try:
            row = conn.execute(
                "SELECT player_id FROM ta_players WHERE (first_name || ' ' || last_name) = ? LIMIT 1",
                (name,),
            ).fetchone()
            if row:
                return row["player_id"]
        except Exception:
            return None
        return None

    @app.get("/api/odds")
    def api_odds():
        """Fetch odds, store raw snapshot, and compute/store ranked candidates.

        v0 behavior:
        - ML (h2h) only candidate generation.
        - Candidate baseline is placeholder until Elo/serve/return/recency overlays are wired.

        Returns both:
        - legacy `outputs` (implied probs from first bookmaker)
        - new `candidates_debug` + `candidates_watchlist` + `candidates_actionable`
        """
        sport_key = request.args.get("sport_key")
        markets = request.args.get("markets", "h2h")
        regions = request.args.get("regions", "us,uk,eu")
        top_n = int(request.args.get("top_n") or os.getenv("TOP_N") or 10)
        refresh_interval_sec = int(os.getenv("REFRESH_INTERVAL_SEC") or 120)
        model_version = os.getenv("MODEL_VERSION")

        if not sport_key:
            return jsonify({"error": "sport_key required"}), 400

        try:
            odds, headers = get_odds(sport_key=sport_key, markets=markets, regions=regions)
            scores, _score_headers = get_scores(sport_key=sport_key, days_from=3)
        except Exception as e:
            msg = str(e)
            hint = "Call /api/tennis_sports to see valid keys (ATP-only use the tennis_atp_* keys)."
            if "OUT_OF_USAGE_CREDITS" in msg or "Usage quota has been reached" in msg:
                hint = "The Odds API key is out of usage credits. Top up / swap the key in Render, or use the latest saved snapshot until credits reset."
            return jsonify({
                "error": "odds_api_error",
                "message": msg,
                "sport_key": sport_key,
                "hint": hint,
            }), 400
        scores_by_id = {s.get("id"): s for s in (scores or []) if s.get("id")}
        surface = infer_surface(sport_key)

        conn = connect()
        ts = utc_now_iso()

        # store legacy raw snapshot (per match)
        for m in odds:
            mid = normalize_match(m)
            conn.execute(
                "INSERT INTO odds_snapshots (ts, sport_key, match_id, payload_json) VALUES (?, ?, ?, ?)",
                (ts, sport_key, mid, json.dumps(m)),
            )

        # store v2 snapshot (one row per refresh)
        cur = conn.execute(
            """
            INSERT INTO snapshots (ts, tour, sport_key, markets, regions, top_n, refresh_interval_sec, model_version, slate_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                "ATP",
                sport_key,
                markets,
                regions,
                top_n,
                refresh_interval_sec,
                model_version,
                None,
            ),
        )
        snapshot_id = cur.lastrowid

        # ---------------- Candidate generation (v0: ML only) ----------------
        candidates = generate_ml_candidates(conn, odds, surface=surface, player_id_lookup=player_id_from_name, top_n=top_n)

        thresholds = _selection_thresholds()

        odds_by_match = {normalize_match(m): m for m in odds}

        for c in candidates:
            if _is_heavy_favorite_ml(c, thresholds):
                c.reasons.append("Heavy-favorite ML: prefer same-player spread / alt market when available; downgrade plain ML unless edge is exceptional.")

        candidates = sorted(candidates, key=lambda c: _selection_sort_key(c, thresholds))

        actionable = []
        watchlist = []
        classified = []
        for c in candidates:
            view_mode, is_actionable = _classify_candidate(c, thresholds)
            alt_market = _best_alternative_market(odds_by_match.get(c.match_id) or {}, c.side) if _is_heavy_favorite_ml(c, thresholds) else None
            strategy_meta = _build_strategy_reasoning(c, "actionable" if is_actionable else view_mode, thresholds, alt_market=alt_market)
            c.selection_tier = strategy_meta["tier"]
            c.units_suggested = strategy_meta["units"]
            c.strategy_summary = strategy_meta["summary"]
            c.strategy_why = strategy_meta["why"]
            c.strategy_risk = strategy_meta["risk"]
            c.alternative_market = strategy_meta["alternative_market"]
            classified.append((c, view_mode, is_actionable))
            if is_actionable:
                actionable.append(c)
            elif view_mode == "watchlist":
                watchlist.append(c)

        def cand_to_dict(c):
            return {
                "match_id": c.match_id,
                "commence_time": c.commence_time,
                "tournament": c.tournament,
                "surface": c.surface or surface,
                "player_a": c.player_a,
                "player_b": c.player_b,
                "market_type": c.market_type,
                "side": c.side,
                "line": c.line,
                "price_decimal": c.price_decimal,
                "price_american": dec_to_american(c.price_decimal),
                "book": c.book,
                "p0": c.p0,
                "p_final": c.p_final,
                "q_implied": c.q_implied,
                "ev": c.ev,
                "confidence": c.confidence,
                "ev_adj": c.ev_adj,
                "matchup_strength": getattr(c, "matchup_strength", None),
                "market_value": getattr(c, "market_value", None),
                "reliability": getattr(c, "reliability", None),
                "component_scores": getattr(c, "component_scores", {}),
                "axis_notes": getattr(c, "axis_notes", {}),
                "reasons": c.reasons,
                "selection_tier": getattr(c, "selection_tier", None),
                "units_suggested": getattr(c, "units_suggested", None),
                "strategy_summary": getattr(c, "strategy_summary", None),
                "strategy_why": getattr(c, "strategy_why", []),
                "strategy_risk": getattr(c, "strategy_risk", []),
                "alternative_market": getattr(c, "alternative_market", None),
            }

        # persist candidates
        created_at = ts
        for c, view_mode, is_actionable in classified:
            conn.execute(
                """
                INSERT INTO ranked_candidates (
                  snapshot_id, created_at,
                  match_id, commence_time, tournament, surface, player_a, player_b,
                  market_type, side, line, price_decimal, price_american, book,
                  p0, p_final, q_implied, ev, confidence, ev_adj,
                  matchup_strength, market_value, reliability,
                  delta_elo_surface, delta_sr, delta_recency, delta_z_raw, delta_z_capped,
                  matchup_flags_json,
                  view_mode, actionable, units_suggested
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    snapshot_id,
                    created_at,
                    c.match_id,
                    c.commence_time,
                    c.tournament,
                    c.surface or surface,
                    c.player_a,
                    c.player_b,
                    c.market_type,
                    c.side,
                    c.line,
                    c.price_decimal,
                    dec_to_american(c.price_decimal),
                    c.book,
                    c.p0,
                    c.p_final,
                    c.q_implied,
                    c.ev,
                    c.confidence,
                    c.ev_adj,
                    getattr(c, "matchup_strength", None),
                    getattr(c, "market_value", None),
                    getattr(c, "reliability", None),
                    None,
                    None,
                    None,
                    None,
                    None,
                    json.dumps({
                        "component_scores": getattr(c, "component_scores", {}),
                        "axis_notes": getattr(c, "axis_notes", {}),
                        "reasons": getattr(c, "reasons", []),
                    }),
                    view_mode,
                    1 if is_actionable else 0,
                    getattr(c, "units_suggested", None),
                ),
            )

        # Optional: capture actionable recommendations into daily_actionables (audit log)
        # Call /api/odds&capture_daily=1 once per day to snapshot the actionable list.
        if (request.args.get("capture_daily") or "0") == "1":
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                date_et = dt.astimezone(ET).date().isoformat()
                created_ts = utc_now_iso()
                rows = conn.execute(
                    """
                    SELECT id as candidate_id, match_id, commence_time, player_a, player_b,
                           market_type, side, line, book, price_decimal, price_american,
                           confidence, ev, ev_adj, view_mode
                    FROM ranked_candidates
                    WHERE snapshot_id = ? AND view_mode IN ('actionable','watchlist')
                    ORDER BY ev_adj DESC NULLS LAST
                    """,
                    (snapshot_id,),
                ).fetchall()

                for r in rows:
                    row_date_et = _et_date_from_iso(r["commence_time"]) or date_et
                    target_table = "daily_actionables" if r["view_mode"] == 'actionable' else "daily_watchlist"
                    conn.execute(
                        f"""
                        INSERT OR IGNORE INTO {target_table} (
                          date_et, created_ts,
                          snapshot_id, candidate_id,
                          sport_key, match_id, commence_time, player_a, player_b,
                          market_type, side, line, book, price_decimal, price_american,
                          confidence, ev, ev_adj,
                          result
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            row_date_et,
                            created_ts,
                            snapshot_id,
                            int(r["candidate_id"]),
                            sport_key,
                            r["match_id"],
                            r["commence_time"],
                            r["player_a"],
                            r["player_b"],
                            r["market_type"],
                            r["side"],
                            r["line"],
                            r["book"],
                            r["price_decimal"],
                            r["price_american"],
                            r["confidence"],
                            r["ev"],
                            r["ev_adj"],
                            "OPEN",
                        ),
                    )
            except Exception:
                pass

        conn.commit()
        conn.close()

        # ---------------- Legacy simple outputs (kept for current UI) ----------------
        outputs = []
        for m in odds:
            mid = normalize_match(m)
            score_obj = scores_by_id.get(m.get("id"))
            bm = (m.get("bookmakers") or [])
            first = bm[0] if bm else None
            h2h = None
            if first:
                for market in first.get("markets") or []:
                    if market.get("key") == "h2h":
                        h2h = market
                        break
            if h2h:
                outs = []
                for o in h2h.get("outcomes") or []:
                    price = o.get("price")
                    prob = (1 / price) if price else None
                    outs.append({
                        "name": o.get("name"),
                        "odds": price,
                        "implied_prob": prob,
                    })

                outputs.append({
                    "match_id": mid,
                    "commence_time": m.get("commence_time"),
                    "home_team": m.get("home_team"),
                    "away_team": m.get("away_team"),
                    "bookmaker": first.get("title") if first else None,
                    "h2h": outs,
                    "completed": (score_obj or {}).get("completed"),
                    "score": (score_obj or {}).get("scores"),
                    "score_last_update": (score_obj or {}).get("last_update"),
                })

        return jsonify({
            "sport_key": sport_key,
            "markets": markets,
            "regions": regions,
            "count": len(odds),
            "requests_remaining": headers.get("x-requests-remaining"),
            "requests_used": headers.get("x-requests-used"),
            "raw": odds,
            "outputs": outputs,
            "ts": ts,
            "snapshot_id": snapshot_id,
            "selection_thresholds": thresholds,
            "candidates_debug": [cand_to_dict(c) for c in candidates],
            "candidates_watchlist": [cand_to_dict(c) for c in watchlist],
            "candidates_actionable": [cand_to_dict(c) for c in actionable],
        })

    return app


def main():
    app = create_app()
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()
