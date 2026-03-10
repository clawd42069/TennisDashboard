from __future__ import annotations

import json
from datetime import datetime, timezone
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
    _settler = {"running": False, "last": None, "last_result": None}

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

    @app.get("/paper")
    def paper():
        """Tab E: paper trading tracker."""
        conn = connect()
        bets = conn.execute(
            "SELECT * FROM paper_bets ORDER BY id DESC LIMIT 200"
        ).fetchall()

        # Build match_id -> label from latest odds snapshot so old rows display nicely
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

        conn.close()

        enriched = []
        for b in bets:
            d = dict(b)
            if not d.get("match_label") and d.get("match_id"):
                d["match_label"] = match_labels.get(d.get("match_id"))
            enriched.append(d)

        # Aggregate performance metrics
        total_bets = len(enriched)
        settled = [b for b in enriched if (b.get("result") in ("WIN", "LOSS", "PUSH"))]
        wins = [b for b in settled if b.get("result") == "WIN"]
        losses = [b for b in settled if b.get("result") == "LOSS"]

        total_pnl = sum((b.get("pnl_dollars") or 0.0) for b in settled)
        total_units = sum((b.get("units") or 0.0) for b in enriched)

        win_rate = (len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) > 0 else None

        # By bet type (market)
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
            "win_rate": win_rate,
            "total_units": total_units,
            "by_type": by_type,
        }

        return render_template("paper.html", bets=enriched, perf=perf)

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
            INSERT INTO paper_bets (ts, match_id, match_label, tournament, player, market, odds_decimal, odds_american, units, note, result, pnl_units, pnl_dollars, settled_ts, source_candidate_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def _settle_open_actionables(conn, date_et: str | None = None, days_from: int = 1):
        """Try to settle OPEN daily actionables using the Odds API scores endpoint.

        Notes:
        - Tennis scores only support a short lookback; default to 1 day.
        - We keep failures soft so the UI can still render.
        """
        days_from = max(1, min(int(days_from or 1), 1))

        sql = "SELECT DISTINCT sport_key FROM daily_actionables WHERE result = 'OPEN' AND sport_key IS NOT NULL"
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

            by_id = {e.get("id"): e for e in (events or []) if e.get("id")}
            row_sql = "SELECT id, match_id, side FROM daily_actionables WHERE result = 'OPEN' AND sport_key = ?"
            row_params = [sport_key]
            if date_et:
                row_sql += " AND date_et = ?"
                row_params.append(date_et)
            rows = conn.execute(row_sql, tuple(row_params)).fetchall()

            for r in rows:
                event = by_id.get(r["match_id"])
                if not event:
                    continue
                if not event.get("completed"):
                    continue
                winner = _infer_winner_name(event)
                if not winner:
                    conn.execute(
                        "UPDATE daily_actionables SET score_json = ? WHERE id = ?",
                        (json.dumps(event), int(r["id"])),
                    )
                    continue
                result = "WIN" if winner == r["side"] else "LOSS"
                conn.execute(
                    "UPDATE daily_actionables SET result = ?, settled_ts = ?, score_json = ? WHERE id = ?",
                    (result, utc_now_iso(), json.dumps(event), int(r["id"])),
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
                if open_n > 0:
                    settle = _settle_open_actionables(conn, days_from=1)
                    conn.commit()
                    _settler["last_result"] = {"open": open_n, **settle}
                else:
                    _settler["last_result"] = {"open": 0, "updated": 0, "errors": []}
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

        return jsonify({
            "snapshot": dict(snap),
            "count": len(rows),
            "candidates": [dict(r) for r in rows],
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
        """Return captured daily actionables for a given ET date.

        Params:
          date: YYYY-MM-DD (defaults to latest date_et)
        """
        date_et = (request.args.get("date") or "").strip() or None
        conn = connect()
        if not date_et:
            row = conn.execute("SELECT date_et FROM daily_actionables ORDER BY date_et DESC LIMIT 1").fetchone()
            date_et = row["date_et"] if row else None
        if not date_et:
            conn.close()
            return jsonify({"date_et": None, "count": 0, "rows": []})

        _settle_open_actionables(conn, date_et=date_et, days_from=1)
        conn.commit()

        rows = conn.execute(
            "SELECT * FROM daily_actionables WHERE date_et = ? ORDER BY ev_adj DESC NULLS LAST",
            (date_et,),
        ).fetchall()
        conn.close()

        now_utc = datetime.now(timezone.utc)
        out = []
        for r in rows:
            d = dict(r)
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

    def _selection_thresholds():
        return {
            "actionable_max_odds": float(os.getenv("ACTIONABLE_MAX_ODDS_DEC") or 5.0),
            "actionable_min_conf": float(os.getenv("ACTIONABLE_MIN_CONF") or 0.66),
            "actionable_min_ev": float(os.getenv("ACTIONABLE_MIN_EV") or 0.03),
            "actionable_min_ev_adj": float(os.getenv("ACTIONABLE_MIN_EV_ADJ") or 0.03),
            "actionable_max_edge": float(os.getenv("ACTIONABLE_MAX_EDGE") or 0.10),
            "watchlist_max_odds": float(os.getenv("WATCHLIST_MAX_ODDS_DEC") or 12.0),
            "watchlist_min_conf": float(os.getenv("WATCHLIST_MIN_CONF") or 0.58),
            "watchlist_min_ev": float(os.getenv("WATCHLIST_MIN_EV") or 0.015),
            "watchlist_min_ev_adj": float(os.getenv("WATCHLIST_MIN_EV_ADJ") or 0.015),
            "heavy_favorite_max_price": float(os.getenv("HEAVY_FAVORITE_MAX_PRICE_DEC") or 1.20),
            "heavy_favorite_min_conf": float(os.getenv("HEAVY_FAVORITE_MIN_CONF") or 0.74),
            "heavy_favorite_min_ev": float(os.getenv("HEAVY_FAVORITE_MIN_EV") or 0.08),
            "heavy_favorite_min_ev_adj": float(os.getenv("HEAVY_FAVORITE_MIN_EV_ADJ") or 0.055),
            "tie_ev_band": float(os.getenv("SELECTION_TIE_EV_BAND") or 0.015),
            "tie_conf_band": float(os.getenv("SELECTION_TIE_CONF_BAND") or 0.04),
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

    def _classify_candidate(c, thresholds: dict) -> tuple[str, bool]:
        if c.price_decimal is None or c.ev is None or c.ev_adj is None or c.q_implied is None or c.p_final is None:
            return "debug", False

        model_edge = abs((c.p_final or 0.0) - (c.q_implied or 0.0))

        actionable = (
            c.price_decimal <= thresholds["actionable_max_odds"]
            and (c.confidence or 0.0) >= thresholds["actionable_min_conf"]
            and c.ev >= thresholds["actionable_min_ev"]
            and c.ev_adj >= thresholds["actionable_min_ev_adj"]
            and model_edge <= thresholds["actionable_max_edge"]
        )
        if actionable:
            if _is_heavy_favorite_ml(c, thresholds):
                heavy_ok = (
                    (c.confidence or 0.0) >= thresholds["heavy_favorite_min_conf"]
                    and c.ev >= thresholds["heavy_favorite_min_ev"]
                    and c.ev_adj >= thresholds["heavy_favorite_min_ev_adj"]
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
        if not latest_ids:
            summary["notes"].append("No snapshots available yet.")
            return summary

        placeholders = ",".join("?" for _ in latest_ids)
        rows = conn.execute(
            f"""
            SELECT price_decimal, confidence, ev, ev_adj, p_final, q_implied
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
            if is_actionable:
                actionables.append(dict(row))
            elif view_mode == "watchlist":
                watchlist.append(dict(row))

        def summarize(items):
            if not items:
                return {"n": 0, "avg_odds": None, "max_odds": None, "avg_conf": None, "avg_ev": None, "avg_ev_adj": None}
            odds = [x["price_decimal"] for x in items if x.get("price_decimal") is not None]
            confs = [x["confidence"] for x in items if x.get("confidence") is not None]
            evs = [x["ev"] for x in items if x.get("ev") is not None]
            ev_adjs = [x["ev_adj"] for x in items if x.get("ev_adj") is not None]
            return {
                "n": len(items),
                "avg_odds": round(sum(odds) / len(odds), 2) if odds else None,
                "max_odds": round(max(odds), 2) if odds else None,
                "avg_conf": round(sum(confs) / len(confs), 3) if confs else None,
                "avg_ev": round(sum(evs) / len(evs), 3) if evs else None,
                "avg_ev_adj": round(sum(ev_adjs) / len(ev_adjs), 3) if ev_adjs else None,
            }

        summary["latest_actionables"] = summarize(actionables)
        summary["latest_watchlist"] = summarize(watchlist)

        def odds_bucket_label(price):
            if price is None:
                return "unknown"
            if price < 1.8:
                return "fav_lt_1.8"
            if price < 2.5:
                return "mid_1.8_2.5"
            if price < 5.0:
                return "dog_2.5_5.0"
            return "longshot_5_plus"

        buckets = {}
        for item in actionables:
            bucket = odds_bucket_label(item.get("price_decimal"))
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
            summary["settled_actionables"]["win_rate"] = round(wins / settled_n, 3)
        else:
            summary["notes"].append("Not enough settled actionables yet for a real threshold backtest.")

        summary["notes"].append("Audit applies the current selection thresholds to the most recent 50 snapshots, even if those snapshots were generated before the new rules shipped.")
        summary["notes"].append("Current actionable design favors lower-variance selections; high-EV fragile dogs are demoted to watchlist.")
        return summary

    def player_id_from_name(conn, name: str | None):
        if not name:
            return None
        # Exact match first
        row = conn.execute(
            "SELECT player_id FROM ta_players WHERE (first_name || ' ' || last_name) = ? LIMIT 1",
            (name,),
        ).fetchone()
        if row:
            return row["player_id"]
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

        for c in candidates:
            if _is_heavy_favorite_ml(c, thresholds):
                c.reasons.append("Heavy-favorite ML: prefer same-player spread / alt market when available; downgrade plain ML unless edge is exceptional.")

        candidates = sorted(candidates, key=lambda c: _selection_sort_key(c, thresholds))

        actionable = []
        watchlist = []
        classified = []
        for c in candidates:
            view_mode, is_actionable = _classify_candidate(c, thresholds)
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
                "reasons": c.reasons,
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
                  delta_elo_surface, delta_sr, delta_recency, delta_z_raw, delta_z_capped,
                  matchup_flags_json,
                  view_mode, actionable, units_suggested
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    view_mode,
                    1 if is_actionable else 0,
                    None,
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
                           confidence, ev, ev_adj
                    FROM ranked_candidates
                    WHERE snapshot_id = ? AND actionable = 1
                    ORDER BY ev_adj DESC NULLS LAST
                    """,
                    (snapshot_id,),
                ).fetchall()

                for r in rows:
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
