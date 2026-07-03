"""Histórico MLB desde la Stats API oficial (statsapi.mlb.com) — GRATIS, sin key.

Una llamada por temporada devuelve TODOS los partidos con linescore (carreras/hits
por equipo Y por entrada — F5 exacto), pitchers probables (también en históricos,
clave para el cerebro matchup) y venue (park factors). Cache-primero en
data/mlb_cache/ (gitignored): re-correr = 0 llamadas.

Cantidades base que produce la tabla (por partido, lado h/a):
  runs, hits, runs_f5 (carreras en las primeras 5 entradas — el mercado F5).

Notas honestas:
- 2020 es la temporada COVID de 60 juegos: se incluye (es data real) — el decay
  temporal de los cerebros ya le baja peso por vieja.
- Los totales de runs incluyen entradas extra (así liquidan los mercados de
  totales); runs_f5 es exacto de las entradas 1-5.
- Partidos sin linescore por entradas (suspendidos/raros) se descartan y cuentan.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MLB_API = "https://statsapi.mlb.com/api/v1"
MLB_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "mlb_cache"
DEFAULT_SEASONS: tuple[int, ...] = tuple(range(2015, 2027))  # 2015..2026
HYDRATE = "linescore,probablePitcher,team,venue"
TIMEOUT_S = 90
DELAY_S = 1.0

_api_calls = 0


def api_calls_made() -> int:
    return _api_calls


def fetch_season(year: int, *, force: bool = False) -> dict:
    """Schedule de la temporada regular completa (1 llamada, cacheada)."""
    global _api_calls
    MLB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = MLB_CACHE_DIR / f"schedule_{year}.json"
    if f.exists() and not force:
        return json.loads(f.read_text(encoding="utf-8"))
    r = requests.get(f"{MLB_API}/schedule", params={
        "sportId": 1, "season": year, "gameType": "R", "hydrate": HYDRATE,
    }, timeout=TIMEOUT_S)
    r.raise_for_status()
    _api_calls += 1
    time.sleep(DELAY_S)
    d = r.json()
    f.write_text(json.dumps(d), encoding="utf-8")
    logger.info("MLB %s: %.1f MB cacheados", year, len(r.content) / 1e6)
    return d


def _game_row(g: dict, season: int) -> dict | None:
    """Una fila normalizada por partido TERMINADO; None si no sirve."""
    if (g.get("status", {}) or {}).get("codedGameState") != "F":
        return None
    ls = g.get("linescore") or {}
    innings = ls.get("innings") or []
    t = ls.get("teams") or {}
    home, away = g["teams"]["home"], g["teams"]["away"]
    th, ta = (t.get("home") or {}), (t.get("away") or {})
    if th.get("runs") is None or ta.get("runs") is None or len(innings) < 5:
        return None
    f5_h = sum(int((i.get("home") or {}).get("runs") or 0) for i in innings[:5])
    f5_a = sum(int((i.get("away") or {}).get("runs") or 0) for i in innings[:5])
    pp_h = home.get("probablePitcher") or {}
    pp_a = away.get("probablePitcher") or {}
    return {
        "date": pd.Timestamp(g["gameDate"][:10]),
        "game_pk": int(g["gamePk"]),
        "home_team": home["team"]["name"],
        "away_team": away["team"]["name"],
        "venue": (g.get("venue") or {}).get("name") or "",
        "runs_h": int(th["runs"]), "runs_a": int(ta["runs"]),
        "hits_h": int(th.get("hits") or 0), "hits_a": int(ta.get("hits") or 0),
        "runs_f5_h": f5_h, "runs_f5_a": f5_a,
        "starter_h_id": pp_h.get("id"), "starter_h": pp_h.get("fullName") or "",
        "starter_a_id": pp_a.get("id"), "starter_a": pp_a.get("fullName") or "",
        "season": str(season),
        "league": "MLB",
        "match_id": str(g["gamePk"]),
    }


def build_mlb_table(
    *, seasons: tuple[int, ...] = DEFAULT_SEASONS, force: bool = False
) -> pd.DataFrame:
    """Tabla consolidada de partidos MLB terminados (cache CSV, re-parseable)."""
    consolidated = MLB_CACHE_DIR / "mlb_games.csv"
    if consolidated.exists() and not force:
        df = pd.read_csv(consolidated, encoding="utf-8")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["season"] = df["season"].astype(str)
        return df
    rows: list[dict] = []
    dropped = 0
    for year in seasons:
        d = fetch_season(year, force=force)
        games = [g for day in d.get("dates", []) for g in day.get("games", [])]
        for g in games:
            row = _game_row(g, year)
            if row is None:
                if (g.get("status", {}) or {}).get("codedGameState") == "F":
                    dropped += 1
                continue
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # doubleheaders: mismo día, mismos equipos — game_pk los distingue; el runner
    # los agrupa por fecha (same-day batching) igual que siempre.
    df = df.drop_duplicates(subset=["game_pk"]).reset_index(drop=True)
    if dropped:
        logger.info("MLB: descartados %d partidos terminados sin linescore completo", dropped)
    MLB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(consolidated, index=False, encoding="utf-8")
    return df
