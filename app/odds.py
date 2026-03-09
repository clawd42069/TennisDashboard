import os
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
API_KEY = os.getenv("ODDS_API_KEY")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _json_or_text(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text


def _ensure_ok(resp: requests.Response):
    if resp.ok:
        return

    payload = _json_or_text(resp)
    headers = dict(resp.headers)
    if isinstance(payload, dict):
        msg = payload.get("message") or payload.get("error") or str(payload)
        code = payload.get("error_code")
        used = headers.get("X-Requests-Used")
        remaining = headers.get("X-Requests-Remaining")
        extra = []
        if code:
            extra.append(f"code={code}")
        if used is not None:
            extra.append(f"used={used}")
        if remaining is not None:
            extra.append(f"remaining={remaining}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        raise RuntimeError(f"Odds API {resp.status_code}: {msg}{suffix}")

    raise RuntimeError(f"Odds API {resp.status_code}: {payload}")


def list_sports():
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    r = requests.get(f"{BASE_URL}/sports", params={"api_key": API_KEY}, timeout=30)
    _ensure_ok(r)
    return r.json(), dict(r.headers)


def get_odds(sport_key: str, regions="us,uk,eu", markets="h2h", odds_format="decimal"):
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    r = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds",
        params={
            "api_key": API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        },
        timeout=30,
    )
    _ensure_ok(r)
    return r.json(), dict(r.headers)


def get_scores(sport_key: str, days_from: int = 3):
    """Fetch live scores/results for a sport_key.

    Note: for tennis, `scores` may be null until match is in-play; still useful for completed + last_update.
    """
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    r = requests.get(
        f"{BASE_URL}/sports/{sport_key}/scores",
        params={
            "api_key": API_KEY,
            "daysFrom": days_from,
        },
        timeout=30,
    )
    _ensure_ok(r)
    return r.json(), dict(r.headers)


def normalize_match(match: dict):
    """Return a stable match id for storage."""
    # The Odds API provides an 'id' field for events in many sports; fallback if absent
    mid = match.get("id")
    if mid:
        return mid
    # fallback: commence + teams
    return f"{match.get('commence_time','')}-{match.get('home_team','')}-{match.get('away_team','')}"
