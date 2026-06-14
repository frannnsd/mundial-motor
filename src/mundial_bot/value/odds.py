"""Cliente de The Odds API (cuotas de la casa) — Agente 3.

The Odds API v4: GET /v4/sports/{sport}/odds?apiKey=..&regions=eu&markets=h2h
- sport key del Mundial: "soccer_fifa_world_cup".
- Cada llamada cuesta `1 × (mercados) × (regiones)` créditos (free tier: 500/mes).
  Por eso pedimos un solo mercado/región por llamada y cacheamos.

El parseo (`parse_events`, `best_1x2`) es puro y se testea con un JSON de ejemplo
sin tocar la red. Para apostar a valor usamos la **mejor cuota disponible** de cada
resultado entre todas las casas (cuanto más alta la cuota, mayor el EV).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import requests

WORLD_CUP_SPORT_KEY = "soccer_fifa_world_cup"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_TIMEOUT_S = 20


@dataclass(frozen=True)
class MatchOdds:
    """Cuotas de un partido, con todas las casas y sus mercados h2h (1X2)."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    # bookmaker -> {nombre_resultado: cuota}
    books_h2h: dict[str, dict[str, float]]

    @property
    def match(self) -> str:
        return f"{self.home_team} vs {self.away_team}"


def parse_events(raw_events: list[dict]) -> list[MatchOdds]:
    """Parsea la respuesta cruda de The Odds API a una lista de MatchOdds (puro)."""
    matches: list[MatchOdds] = []
    for ev in raw_events:
        books: dict[str, dict[str, float]] = {}
        for bk in ev.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {
                    o["name"]: float(o["price"]) for o in market.get("outcomes", [])
                }
                if outcomes:
                    books[bk.get("title", bk.get("key", "?"))] = outcomes
        matches.append(
            MatchOdds(
                event_id=ev.get("id", ""),
                home_team=ev["home_team"],
                away_team=ev["away_team"],
                commence_time=ev.get("commence_time", ""),
                books_h2h=books,
            )
        )
    return matches


def best_1x2(match: MatchOdds) -> dict[str, float]:
    """Mejor cuota (la más alta) de cada resultado 1X2 entre todas las casas.

    Mapea los nombres de equipo a home/away y "Draw" a draw.
    """
    best: dict[str, float] = {}
    for outcomes in match.books_h2h.values():
        for name, price in outcomes.items():
            if name == match.home_team:
                key = "home"
            elif name == match.away_team:
                key = "away"
            else:
                key = "draw"
            if price > best.get(key, 0.0):
                best[key] = price
    return best


def best_book_for(match: MatchOdds, outcome_name: str) -> str | None:
    """Devuelve la casa que ofrece la mejor cuota para un resultado dado."""
    best_price, best_book = 0.0, None
    for book, outcomes in match.books_h2h.items():
        price = outcomes.get(outcome_name, 0.0)
        if price > best_price:
            best_price, best_book = price, book
    return best_book


class OddsClient:
    """Cliente HTTP de The Odds API con cache opcional en disco."""

    def __init__(self, api_key: str, region: str = "eu"):
        self.api_key = api_key
        self.region = region

    def fetch_h2h(
        self, sport_key: str = WORLD_CUP_SPORT_KEY, *, timeout: int = DEFAULT_TIMEOUT_S
    ) -> list[dict]:
        """Trae las cuotas 1X2 crudas del torneo (necesita api_key y red)."""
        if not self.api_key:
            raise RuntimeError("Falta ODDS_API_KEY para consultar The Odds API.")
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_matches(self, sport_key: str = WORLD_CUP_SPORT_KEY) -> list[MatchOdds]:
        """Trae y parsea las cuotas del torneo en una lista de MatchOdds."""
        return parse_events(self.fetch_h2h(sport_key))


def load_sample(path: str | Path) -> list[MatchOdds]:
    """Carga cuotas desde un JSON local (modo offline / testing)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_events(raw)
