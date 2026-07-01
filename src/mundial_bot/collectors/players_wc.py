"""Stats por JUGADOR de los partidos jugados del Mundial 2026 (API-Football).

Baja `/fixtures/players` de cada partido finalizado (FT) del Mundial y arma una
tabla plana con una fila por (fixture, equipo, jugador). Es la materia prima de
la capa de props por jugador (Fase B): shares históricos + minutos.

CACHE PRIMERO (regla dura de rate limits): cada respuesta se guarda en
`data/players_cache/` y NUNCA se re-pide si el JSON ya está en disco.
Una sola llamada por fixture, con delay entre llamadas. El contador módulo
`api_calls_made()` permite auditar cuántas llamadas reales se hicieron.

CONVENCIÓN VERIFICADA EMPÍRICAMENTE: en los conteos de la API, `None` significa 0
(un jugador sin remates trae `shots.total = None`). Acá se convierte SIEMPRE a 0.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from mundial_bot.config import DATA_DIR
from mundial_bot.value.team_aliases import normalize_team

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
TIMEOUT_S = 25
CALL_DELAY_S = 0.4  # respiro entre llamadas reales (rate limit por minuto)

PLAYERS_CACHE_DIR = DATA_DIR / "players_cache"
PLAYER_MATCHES_CSV = PLAYERS_CACHE_DIR / "wc_player_matches.csv"

# Conteos que exporta la tabla (todos con la convención None→0).
COUNT_STATS = (
    "shots", "sot", "goals", "assists", "fouls_committed", "fouls_drawn",
    "yellow", "red", "tackles",
)

_calls_made = 0  # llamadas HTTP reales de este proceso (las de cache no cuentan)


def api_calls_made() -> int:
    """Llamadas HTTP reales hechas por este módulo en el proceso actual."""
    return _calls_made


def _get(key: str, path: str, params: dict) -> dict:
    """GET a API-Football con contador de llamadas (para auditar el presupuesto)."""
    global _calls_made
    r = requests.get(
        f"{API_FOOTBALL_BASE}/{path}",
        headers={"x-apisports-key": key}, params=params, timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    _calls_made += 1
    return r.json()


def _cached_get(key: str, path: str, params: dict, cache_file) -> dict:
    """Lee del cache en disco si existe; si no, llama a la API y guarda el JSON."""
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    raw = _get(key, path, params)
    PLAYERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    time.sleep(CALL_DELAY_S)
    return raw


def fetch_played_fixtures(key: str, *, season: int = DEFAULT_SEASON) -> list[dict]:
    """Partidos FT del Mundial: [{fixture_id, date, home, away}] (equipos normalizados).

    Se cachea a disco (`ft_fixtures.json`): para refrescar la lista tras nuevos
    partidos, borrar ese archivo (los fixture_players ya bajados no se re-piden).
    """
    raw = _cached_get(
        key, "fixtures",
        {"league": WORLD_CUP_LEAGUE_ID, "season": season, "status": "FT"},
        PLAYERS_CACHE_DIR / "ft_fixtures.json",
    )
    out: list[dict] = []
    for item in raw.get("response", []):
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if not fixture.get("id") or not home or not away:
            continue
        out.append({
            "fixture_id": int(fixture["id"]),
            "date": str(fixture.get("date") or "")[:10],
            "home": normalize_team(home),
            "away": normalize_team(away),
        })
    return out


def fetch_played_fixture_ids(key: str, *, season: int = DEFAULT_SEASON) -> list[int]:
    """IDs de los partidos finalizados (FT) del Mundial."""
    return [f["fixture_id"] for f in fetch_played_fixtures(key, season=season)]


def fetch_upcoming_fixtures(key: str, *, season: int = DEFAULT_SEASON) -> list[dict]:
    """Partidos PRÓXIMOS (NS) del Mundial, ordenados por fecha, mismo formato que FT.

    Cache en `ns_fixtures.json` (borrar para refrescar la lista tras cada jornada).
    """
    raw = _cached_get(
        key, "fixtures",
        {"league": WORLD_CUP_LEAGUE_ID, "season": season, "status": "NS"},
        PLAYERS_CACHE_DIR / "ns_fixtures.json",
    )
    out: list[dict] = []
    for item in raw.get("response", []):
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if not fixture.get("id") or not home or not away:
            continue
        out.append({
            "fixture_id": int(fixture["id"]),
            "date": str(fixture.get("date") or ""),
            "home": normalize_team(home),
            "away": normalize_team(away),
        })
    return sorted(out, key=lambda f: f["date"])


def fetch_fixture_players(key: str, fixture_id: int) -> dict:
    """Stats por jugador de un fixture, con cache JSON en disco (1 llamada máx)."""
    return _cached_get(
        key, "fixtures/players", {"fixture": fixture_id},
        PLAYERS_CACHE_DIR / f"fixture_players_{fixture_id}.json",
    )


def fetch_lineups(key: str, fixture_id: int) -> dict:
    """Alineaciones (XI confirmado + suplentes) de un fixture, con cache en disco.

    OJO LEAKAGE: la API publica esto ~20-40 min antes del kickoff. Solo puede
    alimentar predicciones POSTERIORES a su publicación (ver players/props.py).
    """
    return _cached_get(
        key, "fixtures/lineups", {"fixture": fixture_id},
        PLAYERS_CACHE_DIR / f"lineups_{fixture_id}.json",
    )


def _n(v) -> int:
    """Conteo de la API → int con la convención None→0 (verificada empíricamente)."""
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def parse_fixture_players(raw: dict, *, fixture_id: int, date: str = "") -> list[dict]:
    """Aplana la respuesta de /fixtures/players: una fila por (equipo, jugador). Puro.

    Aplica None→0 en todos los conteos y en minutos. `substitute` (bool) se
    conserva porque alimenta los minutos esperados de titulares vs suplentes.
    """
    rows: list[dict] = []
    for side in raw.get("response", []) or []:
        team = normalize_team((side.get("team") or {}).get("name") or "")
        for entry in side.get("players", []) or []:
            player = entry.get("player") or {}
            stats = (entry.get("statistics") or [{}])[0]
            games = stats.get("games") or {}
            shots = stats.get("shots") or {}
            goals = stats.get("goals") or {}
            fouls = stats.get("fouls") or {}
            cards = stats.get("cards") or {}
            tackles = stats.get("tackles") or {}
            rows.append({
                "fixture_id": fixture_id,
                "date": date,
                "team": team,
                "player_id": _n(player.get("id")),
                "player_name": player.get("name") or "",
                "position": (games.get("position") or "").strip().upper()[:1],
                "substitute": bool(games.get("substitute")),
                "minutes": _n(games.get("minutes")),
                "shots": _n(shots.get("total")),
                "sot": _n(shots.get("on")),
                "goals": _n(goals.get("total")),
                "assists": _n(goals.get("assists")),
                "fouls_committed": _n(fouls.get("committed")),
                "fouls_drawn": _n(fouls.get("drawn")),
                "yellow": _n(cards.get("yellow")),
                "red": _n(cards.get("red")),
                "tackles": _n(tackles.get("total")),
            })
    return rows


def build_player_match_table(key: str, *, refresh: bool = False) -> pd.DataFrame:
    """Tabla (fixture, equipo, jugador) de TODOS los partidos FT del Mundial.

    Cache en dos niveles: el CSV consolidado (`wc_player_matches.csv`) se lee
    directo si existe (salvo `refresh=True`); y aun refrescando, cada fixture ya
    bajado se lee de su JSON en disco — refrescar solo agrega los partidos nuevos.
    """
    if PLAYER_MATCHES_CSV.exists() and not refresh:
        return pd.read_csv(PLAYER_MATCHES_CSV, encoding="utf-8")

    rows: list[dict] = []
    for fx in fetch_played_fixtures(key):
        raw = fetch_fixture_players(key, fx["fixture_id"])
        rows.extend(parse_fixture_players(raw, fixture_id=fx["fixture_id"], date=fx["date"]))
    df = pd.DataFrame(rows)
    if not df.empty:
        PLAYERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PLAYER_MATCHES_CSV, index=False, encoding="utf-8")
    return df
