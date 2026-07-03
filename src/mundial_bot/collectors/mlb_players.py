"""Gamelogs por jugador y lineups desde la MLB Stats API (gratis, sin key).

Insumos de la capa de props (players/mlb_props.py):
- gameLog del pitcher (group=pitching): Ks, innings, bateadores enfrentados por start.
- gameLog del bateador (group=hitting): hits, HR, AB, PA por partido.
- boxscore de un juego: lineup titular (orden de bateo 1..9) por lado.

Cache-primero en data/mlb_cache/players/ (gitignored, misma raíz que mlb_data):
si el JSON existe, 0 llamadas. Delay 0.3 s entre llamadas reales — la API es
gratis pero no abusamos.

Nota sobre inningsPitched: la notación "5.2" son 5 entradas y DOS TERCIOS
(los decimales .1/.2 cuentan outs, no décimas). Acá se convierte SIEMPRE a
outs enteros y de ahí a innings reales = outs/3.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from mundial_bot.collectors.mlb_data import MLB_API, MLB_CACHE_DIR

logger = logging.getLogger(__name__)

PLAYERS_CACHE_DIR = MLB_CACHE_DIR / "players"
TIMEOUT_S = 30
DELAY_S = 0.3

_api_calls = 0


def api_calls_made() -> int:
    """Llamadas HTTP reales hechas por este módulo (cache hits no cuentan)."""
    return _api_calls


def ip_to_outs(innings_pitched: str | float | None) -> int:
    """Convierte inningsPitched notación MLB a outs: "5.2" = 5⅔ entradas → 17 outs.

    El dígito después del punto son TERCIOS de entrada (0, 1 o 2), no décimas.
    """
    if innings_pitched is None:
        return 0
    whole, _, frac = str(innings_pitched).partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def _get_cached(cache_file: Path, url: str, params: dict, *, force: bool) -> dict:
    """GET cacheado a JSON: si el archivo existe (y no force), 0 llamadas."""
    global _api_calls
    PLAYERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    r = requests.get(url, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    _api_calls += 1
    time.sleep(DELAY_S)
    d = r.json()
    cache_file.write_text(json.dumps(d), encoding="utf-8")
    return d


def _gamelog_splits(person_id: int, season: int, group: str, *, force: bool) -> list[dict]:
    """Splits crudos del gameLog de una persona/temporada ([] si no jugó)."""
    kind = "pitcher" if group == "pitching" else "batter"
    f = PLAYERS_CACHE_DIR / f"{kind}_{person_id}_{season}.json"
    d = _get_cached(f, f"{MLB_API}/people/{person_id}/stats", {
        "stats": "gameLog", "season": season, "group": group,
    }, force=force)
    stats = d.get("stats") or []
    return (stats[0].get("splits") or []) if stats else []


def fetch_pitcher_gamelog(person_id: int, season: int, *, force: bool = False) -> list[dict]:
    """gameLog de pitcheo: una fila por aparición, con innings ya convertidos.

    Campos por aparición: date, game_pk, is_start, strikeouts (int),
    outs (int), innings (float = outs/3), batters_faced, hits, home_runs,
    earned_runs. Incluye relevos (is_start=False) — la capa de props filtra
    solo los starts.
    """
    out: list[dict] = []
    for sp in _gamelog_splits(person_id, season, "pitching", force=force):
        st = sp.get("stat") or {}
        outs = ip_to_outs(st.get("inningsPitched"))
        out.append({
            "date": sp.get("date"),
            "game_pk": (sp.get("game") or {}).get("gamePk"),
            "is_start": int(st.get("gamesStarted") or 0) > 0,
            "strikeouts": int(st.get("strikeOuts") or 0),
            "outs": outs,
            "innings": outs / 3.0,
            "batters_faced": int(st.get("battersFaced") or 0),
            "hits": int(st.get("hits") or 0),
            "home_runs": int(st.get("homeRuns") or 0),
            "earned_runs": int(st.get("earnedRuns") or 0),
        })
    return out


def fetch_batter_gamelog(person_id: int, season: int, *, force: bool = False) -> list[dict]:
    """gameLog de bateo: date, game_pk, hits, home_runs, at_bats, plate_appearances."""
    out: list[dict] = []
    for sp in _gamelog_splits(person_id, season, "hitting", force=force):
        st = sp.get("stat") or {}
        out.append({
            "date": sp.get("date"),
            "game_pk": (sp.get("game") or {}).get("gamePk"),
            "hits": int(st.get("hits") or 0),
            "home_runs": int(st.get("homeRuns") or 0),
            "at_bats": int(st.get("atBats") or 0),
            "plate_appearances": int(st.get("plateAppearances") or 0),
        })
    return out


def fetch_game_lineup(game_pk: int, *, force: bool = False) -> dict[str, list[dict]]:
    """Lineup TITULAR (orden de bateo 1..9) por lado desde el boxscore.

    battingOrder viene como string "100".."900" para titulares; los que entran
    después llevan sufijo ("101", "402", ...) y se excluyen. Devuelve
    {"home": [...], "away": [...]} con person_id, full_name, batting_order,
    ordenado por batting_order.
    """
    f = PLAYERS_CACHE_DIR / f"lineup_{game_pk}.json"
    d = _get_cached(f, f"{MLB_API}/game/{game_pk}/boxscore", {}, force=force)
    out: dict[str, list[dict]] = {}
    for side in ("home", "away"):
        players = ((d.get("teams") or {}).get(side) or {}).get("players") or {}
        starters: list[dict] = []
        for p in players.values():
            bo_raw = p.get("battingOrder")
            if not bo_raw:
                continue
            bo = int(bo_raw)
            if bo % 100 != 0:  # sustituto: entró en ese slot, no arrancó
                continue
            person = p.get("person") or {}
            starters.append({
                "person_id": int(person.get("id")),
                "full_name": person.get("fullName") or "",
                "batting_order": bo // 100,
            })
        out[side] = sorted(starters, key=lambda x: x["batting_order"])
    return out
