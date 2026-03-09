from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
            # import ATP + build features (includes surface Elo build)
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

    @app.post("/paper/add")
    def paper_add():
        odds_dec = request.form.get("odds_decimal")
        odds_dec_f = float(odds_dec) if odds_dec not in (None, "") else None
        odds_am = request.form.get("odds_american")
        odds_am_i = int(odds_am) if odds_am not in (None, "") else None
        if odds_am_i is None and odds_dec_f is not None:
            odds_am_i = dec_to_american(odds_dec_f)

        units = float(request.form.get("units") or 0)
        result = request.form.get("result") or "OPEN"
        pnl_units = pnl_units_for_result(result, units, odds_am_i)
        pnl_dollars = units_to_dollars(pnl_units, 500.0)

        conn = connect()
        conn.execute(
            """
            INSERT INTO paper_bets (ts, match_id, match_label, tournament, player, market, odds_decimal, odds_american, units, note, result, pnl_units, pnl_dollars, settled_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                request.form.get("match_id") or None,
                request.form.get("match_label") or None,
                request.form.get("tournament") or None,
                request.form.get("player") or "",
                request.form.get("market") or "h2h",
                odds_dec_f,
                odds_am_i,
                units,
                request.form.get("note") or None,
                result,
                pnl_units,
                pnl_dollars,
                utc_now_iso() if result in ("WIN", "LOSS", "PUSH") else None,
            ),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("paper"))

    @app.get("/strategies")
    def strategies():
        """Tab D: strategies + backtests (stub until historical DB is integrated)."""
        conn = connect()
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY id DESC LIMIT 200"
        ).fetchall()
        conn.close()
        return render_template("strategies.html", strategies=rows)

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
        """Fetch latest ranked candidates from DB (no Odds API call).

        Params:
          view: actionable|debug|all (default all)
          limit: int (default 50)
        """
        view = (request.args.get("view") or "all").lower()
        limit = int(request.args.get("limit") or 50)
        limit = max(1, min(500, limit))

        conn = connect()
        snap = conn.execute("SELECT id, ts, sport_key, markets, regions, top_n, model_version FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if not snap:
            conn.close()
            return jsonify({"snapshot": None, "candidates": [], "note": "No snapshots yet. Call /api/odds first."})

        where = ""
        params = [snap["id"]]
        if view == "actionable":
            where = " AND actionable = 1"
        elif view == "debug":
            where = ""
        elif view == "all":
            where = ""
        else:
            conn.close()
            return jsonify({"error": "invalid view (use actionable|debug|all)"}), 400

        rows = conn.execute(
            """
            SELECT *
            FROM ranked_candidates
            WHERE snapshot_id = ?
            """ + where + "\nORDER BY ev_adj DESC NULLS LAST, confidence DESC NULLS LAST\nLIMIT ?",
            (*params, limit),
        ).fetchall()
        conn.close()

        return jsonify({
            "snapshot": dict(snap),
            "count": len(rows),
            "candidates": [dict(r) for r in rows],
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

        rows = conn.execute(
            "SELECT * FROM daily_actionables WHERE date_et = ? ORDER BY ev_adj DESC NULLS LAST",
            (date_et,),
        ).fetchall()
        conn.close()
        return jsonify({"date_et": date_et, "count": len(rows), "rows": [dict(r) for r in rows]})

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
        - new `candidates_debug` + `candidates_actionable`
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
            return jsonify({
                "error": "odds_api_error",
                "message": str(e),
                "sport_key": sport_key,
                "hint": "Call /api/tennis_sports to see valid keys (ATP-only use the tennis_atp_* keys).",
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

        # Actionable filter (happy-medium defaults):
        # - avoid ultra-longshots until we have richer features + backtests
        # - require real edge (EV) AND decent confidence
        MAX_ODDS_DEC = float(os.getenv("ACTIONABLE_MAX_ODDS_DEC") or 12.0)
        MIN_CONF = float(os.getenv("ACTIONABLE_MIN_CONF") or 0.62)
        MIN_EV = float(os.getenv("ACTIONABLE_MIN_EV") or 0.02)  # +2% per 1u risk
        MIN_EV_ADJ = float(os.getenv("ACTIONABLE_MIN_EV_ADJ") or 0.02)

        actionable = []
        for c in candidates:
            if c.price_decimal is None or c.ev is None or c.ev_adj is None:
                continue
            if c.price_decimal > MAX_ODDS_DEC:
                continue
            if (c.confidence or 0) < MIN_CONF:
                continue
            if c.ev < MIN_EV:
                continue
            if c.ev_adj < MIN_EV_ADJ:
                continue
            actionable.append(c)

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
        for c in candidates:
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
                    "debug",
                    1 if c in actionable else 0,
                    None,
                ),
            )

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
            "candidates_debug": [cand_to_dict(c) for c in candidates],
            "candidates_actionable": [cand_to_dict(c) for c in actionable],
        })

    return app


def main():
    app = create_app()
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()
