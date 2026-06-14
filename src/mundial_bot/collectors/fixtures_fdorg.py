"""Colector de fixtures del Mundial desde football-data.org — Agente 1 (alternativa free).

El tier gratis de API-Football NO da acceso a la temporada 2026. football-data.org
sí incluye el Mundial en su tier gratis (10 req/min) y trae el árbitro asignado.

football-data.org v4:
  GET /v4/competitions/WC/matches?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD
  header: X-Auth-Token. Competición "WC" = FIFA World Cup.
"""

from __future__ import annotations

import requests

from mundial_bot.collectors.fixtures import Fixture

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
WORLD_CUP_CODE = "WC"
TIMEOUT_S = 20


def parse_fdorg_matches(raw: dict) -> list[Fixture]:
    """Parsea la respuesta de football-data.org a una lista de Fixture (puro)."""
    out: list[Fixture] = []
    for m in raw.get("matches", []):
        home = (m.get("homeTeam") or {}).get("name")
        away = (m.get("awayTeam") or {}).get("name")
        if not home or not away:
            continue
        referees = m.get("referees") or []
        referee = referees[0].get("name") if referees else None
        out.append(
            Fixture(
                home_team=home,
                away_team=away,
                date=m.get("utcDate", "") or "",
                referee=referee,
                round=m.get("stage", "") or "",
            )
        )
    return out


class FootballDataClient:
    """Cliente de fixtures de football-data.org."""

    def __init__(self, token: str):
        self.token = token

    def fetch(self, *, date_from: str, date_to: str, timeout: int = TIMEOUT_S) -> dict:
        """Trae los partidos del Mundial entre dos fechas (necesita token y red)."""
        if not self.token:
            raise RuntimeError("Falta FOOTBALL_DATA_KEY para consultar football-data.org.")
        resp = requests.get(
            f"{FOOTBALL_DATA_BASE}/competitions/{WORLD_CUP_CODE}/matches",
            params={"dateFrom": date_from, "dateTo": date_to},
            headers={"X-Auth-Token": self.token},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_fixtures(self, *, date: str) -> list[Fixture]:
        """Trae y parsea los partidos de una fecha (de un solo día)."""
        return parse_fdorg_matches(self.fetch(date_from=date, date_to=date))
