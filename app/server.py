import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import Flask, render_template, request, redirect, url_for, jsonify

import math

from .db import migrate, connect
from .odds import list_sports, get_odds, get_scores, normalize_match

import os

APP_PORT = int(os.getenv("PORT", "8008"))


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def create_app():
    app = Flask(__name__)
    migrate()

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

    @app.get("/api/tennis_sports")
    def api_tennis_sports():
        sports, headers = list_sports()
        tennis = [s for s in sports if (s.get("group") == "Tennis") or ("tennis" in (s.get("key") or ""))]
        return jsonify({
            "tennis": tennis,
            "requests_remaining": headers.get("x-requests-remaining"),
            "requests_used": headers.get("x-requests-used"),
        })

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

    @app.get("/api/odds")
    def api_odds():
        sport_key = request.args.get("sport_key")
        markets = request.args.get("markets", "h2h")
        if not sport_key:
            return jsonify({"error": "sport_key required"}), 400
        odds, headers = get_odds(sport_key=sport_key, markets=markets)
        scores, _score_headers = get_scores(sport_key=sport_key, days_from=3)
        scores_by_id = {s.get("id"): s for s in (scores or []) if s.get("id")}

        # store snapshot
        conn = connect()
        ts = utc_now_iso()
        for m in odds:
            mid = normalize_match(m)
            conn.execute(
                "INSERT INTO odds_snapshots (ts, sport_key, match_id, payload_json) VALUES (?, ?, ?, ?)",
                (ts, sport_key, mid, json.dumps(m)),
            )
        conn.commit()
        conn.close()

        # compute simple outputs: implied probs from first bookmaker h2h
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
                    "note": "Scores come from The Odds API /scores endpoint when available. Style/progression/fatigue gets wired once Tennis Abstract DB is imported.",
                })

        return jsonify({
            "sport_key": sport_key,
            "markets": markets,
            "count": len(odds),
            "requests_remaining": headers.get("x-requests-remaining"),
            "requests_used": headers.get("x-requests-used"),
            "raw": odds,
            "outputs": outputs,
            "ts": ts,
        })

    return app


def main():
    app = create_app()
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()
