"""Colector de fixtures del Mundial 2026 desde API-Football — Agente 1.

Trae los partidos reales del día (equipos, fecha, **árbitro** y ronda) para que el
predictor no use la lista de ejemplo. El árbitro alimenta el modelo de tarjetas y la
ronda determina si es eliminación directa (más tarjetas).

API-Football: GET /fixtures?league=1&season=2026&date=YYYY-MM-DD
  header: x-apisports-key. Free tier: 100 req/día. league=1 = "World Cup".
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
TIMEOUT_S = 20


@dataclass(frozen=True)
class Fixture:
    home_team: str
    away_team: str
    date: str
    referee: str | None = None
    round: str = ""

    @property
    def match(self) -> str:
        return f"{self.home_team} vs {self.away_team}"

    @property
    def knockout(self) -> bool:
        """True si es eliminación directa (octavos en adelante)."""
        r = self.round.lower()
        return bool(r) and "group" not in r and "qualif" not in r


def parse_fixtures(raw: dict) -> list[Fixture]:
    """Parsea la respuesta de API-Football a una lista de Fixture (puro)."""
    out: list[Fixture] = []
    for item in raw.get("response", []):
        teams = item.get("teams", {})
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if not home or not away:
            continue
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        out.append(
            Fixture(
                home_team=home,
                away_team=away,
                date=fixture.get("date", ""),
                referee=fixture.get("referee"),
                round=league.get("round", "") or "",
            )
        )
    return out


class FixturesClient:
    """Cliente de fixtures de API-Football."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(
        self,
        *,
        date: str | None = None,
        season: int = DEFAULT_SEASON,
        league: int = WORLD_CUP_LEAGUE_ID,
        timeout: int = TIMEOUT_S,
    ) -> dict:
        """Trae los fixtures crudos (necesita api_key y red)."""
        if not self.api_key:
            raise RuntimeError("Falta API_FOOTBALL_KEY para consultar API-Football.")
        params: dict[str, str | int] = {"league": league, "season": season}
        if date:
            params["date"] = date
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            params=params,
            headers={"x-apisports-key": self.api_key},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_fixtures(
        self, *, date: str | None = None, season: int = DEFAULT_SEASON
    ) -> list[Fixture]:
        """Trae y parsea los fixtures de una fecha."""
        return parse_fixtures(self.fetch(date=date, season=season))
