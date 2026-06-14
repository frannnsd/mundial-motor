"""Colector de forma reciente por equipo desde API-Football Pro — Agente 1 (cerebro).

Para cada equipo del Mundial 2026 baja sus últimos N partidos y, de cada uno, las
estadísticas reales: córners (a favor y en contra), tarjetas, tiros, faltas + árbitro.
Es muchísimo mejor que los 314 partidos históricos de StatsBomb porque captura la
**forma reciente** de los 48 equipos que realmente juegan.

Produce las MISMAS columnas que el colector de StatsBomb (match_id, team, opponent,
corners_for, corners_against, cards, fouls, referee) — más `shots` — así los modelos
de córners y tarjetas lo consumen sin cambios.

OJO: hace ~1 request por partido (cientos en total). Cacheá el resultado.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from mundial_bot.config import CACHE_DIR
from mundial_bot.value.team_aliases import normalize_team

logger = logging.getLogger(__name__)

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
TEAM_STATS_CACHE = CACHE_DIR / "team_match_stats.csv"
TIMEOUT_S = 25


def _num(value) -> int:
    """Convierte un valor de estadística (puede venir None, '12', '55%') a entero."""
    if value in (None, ""):
        return 0
    if isinstance(value, str) and value.endswith("%"):
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _extract(stat_list: list[dict]) -> dict[str, int]:
    """Saca córners/tarjetas/tiros/faltas de la lista de estadísticas de un equipo."""
    d = {s.get("type"): s.get("value") for s in stat_list}
    return {
        "corners": _num(d.get("Corner Kicks")),
        "cards": _num(d.get("Yellow Cards")) + _num(d.get("Red Cards")),
        "shots": _num(d.get("Total Shots")),
        "fouls": _num(d.get("Fouls")),
    }


def _get(key: str, path: str, params: dict) -> dict:
    resp = requests.get(
        f"{API_FOOTBALL_BASE}{path}",
        headers={"x-apisports-key": key},
        params=params,
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def get_wc_team_ids(key: str, *, season: int = DEFAULT_SEASON) -> dict[int, str]:
    """Devuelve {team_id: nombre} de los equipos del Mundial."""
    raw = _get(key, "/teams", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
    out: dict[int, str] = {}
    for item in raw.get("response", []):
        team = item.get("team", {})
        if team.get("id") and team.get("name"):
            out[int(team["id"])] = team["name"]
    return out


def collect_team_stats(key: str, *, last: int = 12, season: int = DEFAULT_SEASON) -> pd.DataFrame:
    """Baja los últimos `last` partidos de cada equipo del Mundial con sus estadísticas."""
    teams = get_wc_team_ids(key, season=season)
    logger.info("Equipos del Mundial: %d", len(teams))
    seen_fixtures: set[int] = set()
    rows: list[dict] = []

    for tid in teams:
        try:
            fixtures = _get(key, "/fixtures", {"team": tid, "last": last}).get("response", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin fixtures para team %s: %s", tid, exc)
            continue

        for fx in fixtures:
            fixture = fx.get("fixture", {})
            fid = fixture.get("id")
            if not fid or fid in seen_fixtures:
                continue
            seen_fixtures.add(fid)
            fts = fx.get("teams", {})
            home_id = (fts.get("home") or {}).get("id")
            id_to_name = {
                (fts.get("home") or {}).get("id"): (fts.get("home") or {}).get("name"),
                (fts.get("away") or {}).get("id"): (fts.get("away") or {}).get("name"),
            }
            try:
                stats_raw = _get(key, "/fixtures/statistics", {"fixture": fid}).get("response", [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sin stats para fixture %s: %s", fid, exc)
                continue
            if len(stats_raw) < 2:
                continue

            per_team = {
                entry.get("team", {}).get("id"): _extract(entry.get("statistics", []))
                for entry in stats_raw
            }
            ids = [i for i in per_team if i is not None]
            if len(ids) != 2:
                continue

            referee = fixture.get("referee")
            for tid_a, tid_b in (ids, ids[::-1]):
                a, b = per_team[tid_a], per_team[tid_b]
                match_date = pd.to_datetime(
                    fixture.get("date"), utc=True, errors="coerce"
                )
                if match_date is not pd.NaT:
                    match_date = match_date.tz_localize(None)
                rows.append({
                    "match_id": int(fid),
                    "date": match_date,
                    "team": normalize_team(id_to_name.get(tid_a, "")),
                    "opponent": normalize_team(id_to_name.get(tid_b, "")),
                    "corners_for": a["corners"],
                    "corners_against": b["corners"],
                    "cards": a["cards"],
                    "fouls": a["fouls"],
                    "shots": a["shots"],
                    "referee": referee,
                    "is_home": int(tid_a == home_id),
                })

    return pd.DataFrame(rows)


def build_cache(key: str, *, last: int = 12) -> Path:
    """Baja y cachea las estadísticas a CSV. Devuelve la ruta."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_team_stats(key, last=last)
    df.to_csv(TEAM_STATS_CACHE, index=False, encoding="utf-8")
    return TEAM_STATS_CACHE


def load_team_stats() -> pd.DataFrame:
    """Carga las estadísticas cacheadas (error si no existen)."""
    if not TEAM_STATS_CACHE.exists():
        raise FileNotFoundError("Cache de team stats ausente. Corré fetch_team_stats.py.")
    return pd.read_csv(TEAM_STATS_CACHE, encoding="utf-8")
