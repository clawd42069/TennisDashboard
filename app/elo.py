"""Elo utilities + ATP surface Elo cache builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .modeling import elo_to_p


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EloRow:
    elo: float
    matches: int


def _k_factor(matches_played: int) -> float:
    """Simple diminishing K-factor."""
    if matches_played < 20:
        return 32.0
    if matches_played < 50:
        return 24.0
    return 16.0


def build_atp_surface_elo(conn, since_year: int = 2015) -> None:
    """(Re)build ATP surface Elo table from Tennis Abstract match history."""
    # Pull minimal match rows in chronological order
    rows = conn.execute(
        """
        SELECT tourney_date, surface, winner_id, loser_id
        FROM ta_matches
        WHERE tourney_date >= ?
          AND surface IS NOT NULL AND surface != ''
          AND winner_id IS NOT NULL AND loser_id IS NOT NULL
        ORDER BY tourney_date ASC, match_id ASC
        """,
        (f"{since_year}0101",),
    ).fetchall()

    surfaces = {"Hard", "Clay", "Grass"}
    # ratings[surface][player_id] = EloRow
    ratings: dict[str, dict[int, EloRow]] = {s: {} for s in surfaces}

    def get_row(surf: str, pid: int) -> EloRow:
        d = ratings[surf]
        r = d.get(pid)
        if r is None:
            r = EloRow(elo=1500.0, matches=0)
            d[pid] = r
        return r

    for r in rows:
        surf = r["surface"]
        if surf not in surfaces:
            # Ignore Carpet/other
            continue
        w = int(r["winner_id"])
        l = int(r["loser_id"])
        rw = get_row(surf, w)
        rl = get_row(surf, l)

        p = elo_to_p(rw.elo - rl.elo)
        kw = _k_factor(rw.matches)
        kl = _k_factor(rl.matches)
        # Winner gets (1-p), loser gets (0-p)
        rw.elo = rw.elo + kw * (1.0 - p)
        rl.elo = rl.elo + kl * (0.0 - (1.0 - p))
        rw.matches += 1
        rl.matches += 1

    # Write table
    ts = utc_now_iso()
    conn.execute("DELETE FROM atp_surface_elo")
    for surf, d in ratings.items():
        for pid, er in d.items():
            conn.execute(
                "INSERT OR REPLACE INTO atp_surface_elo (player_id, surface, elo, matches, updated_ts) VALUES (?, ?, ?, ?, ?)",
                (pid, surf, float(er.elo), int(er.matches), ts),
            )
    conn.commit()


def ensure_atp_surface_elo(conn, max_age_hours: int = 24) -> None:
    """Ensure surface Elo cache exists and is reasonably fresh."""
    row = conn.execute("SELECT updated_ts FROM atp_surface_elo LIMIT 1").fetchone()
    if not row:
        build_atp_surface_elo(conn)
        return
    try:
        updated = datetime.fromisoformat(row["updated_ts"].replace("Z", "+00:00"))
    except Exception:
        build_atp_surface_elo(conn)
        return

    age = datetime.now(timezone.utc) - updated
    if age.total_seconds() > max_age_hours * 3600:
        build_atp_surface_elo(conn)


def get_player_surface_elo(conn, player_id: int | None, surface: str | None) -> tuple[float | None, int | None]:
    if not player_id or not surface:
        return None, None
    r = conn.execute(
        "SELECT elo, matches FROM atp_surface_elo WHERE player_id = ? AND surface = ?",
        (player_id, surface),
    ).fetchone()
    if not r:
        return None, None
    return float(r["elo"]), int(r["matches"])
