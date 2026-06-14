"""Resultados finalizados del Mundial 2026 desde API-Football, en formato martj42.

Esto es lo que hace que el Elo se **autoalimente**: a medida que se juegan partidos
del Mundial, sus resultados se suman al entrenamiento del Elo, así los ratings
reflejan lo que está pasando AHORA (martj42 tarda en incorporar el torneo en curso).

Devuelve las mismas columnas que el dataset martj42 (date, home_team, away_team,
home_score, away_score, tournament, neutral), con los nombres mapeados al modelo.
"""

from __future__ import annotations

import pandas as pd
import requests

from mundial_bot.config import CACHE_DIR
from mundial_bot.value.team_aliases import normalize_team

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
WC_RESULTS_CACHE = CACHE_DIR / "wc2026_results.csv"
TIMEOUT_S = 25


def parse_wc_results(raw: dict) -> pd.DataFrame:
    """Convierte la respuesta de API-Football a formato martj42 (solo partidos FT). Puro."""
    rows: list[dict] = []
    for item in raw.get("response", []):
        fixture = item.get("fixture", {})
        if (fixture.get("status", {}) or {}).get("short") != "FT":
            continue
        goals = item.get("goals", {})
        home_score, away_score = goals.get("home"), goals.get("away")
        if home_score is None or away_score is None:
            continue
        teams = item.get("teams", {})
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if not home or not away:
            continue
        rows.append({
            "date": pd.to_datetime(fixture.get("date"), utc=True).tz_localize(None).normalize(),
            "home_team": normalize_team(home),
            "away_team": normalize_team(away),
            "home_score": int(home_score),
            "away_score": int(away_score),
            "tournament": "FIFA World Cup",
            "neutral": True,
        })
    return pd.DataFrame(rows)


def fetch_wc_results(key: str, *, season: int = DEFAULT_SEASON) -> pd.DataFrame:
    """Baja los partidos finalizados del Mundial y los devuelve en formato martj42."""
    raw = requests.get(
        f"{API_FOOTBALL_BASE}/fixtures", headers={"x-apisports-key": key},
        params={"league": WORLD_CUP_LEAGUE_ID, "season": season}, timeout=TIMEOUT_S,
    ).json()
    return parse_wc_results(raw)


def build_cache(key: str) -> pd.DataFrame:
    """Baja los resultados del Mundial y los cachea. Devuelve el DataFrame."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_wc_results(key)
    df.to_csv(WC_RESULTS_CACHE, index=False, encoding="utf-8")
    return df


def load_wc_results() -> pd.DataFrame:
    """Carga los resultados cacheados del Mundial (DataFrame vacío si no hay)."""
    if not WC_RESULTS_CACHE.exists():
        return pd.DataFrame()
    df = pd.read_csv(WC_RESULTS_CACHE, encoding="utf-8")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
