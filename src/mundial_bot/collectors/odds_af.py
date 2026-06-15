"""Lector de cuotas EN VIVO desde API-Football Pro (/odds) — todas las casas.

Lee las cuotas de 265+ casas por partido y se queda con la MEJOR (la más alta) de
cada resultado — porque cuanto más alta la cuota, mejor pagás. Cubre 1X2, over/under
de goles, ambos marcan, córners, tarjetas, etc.

El evaluador después compara estas cuotas contra la probabilidad del modelo para
decir cuáles son BUENAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
TIMEOUT_S = 25


@dataclass
class MarketOdds:
    """Mejor cuota de cada resultado de un mercado, con la casa que la da."""

    market: str
    best: dict[str, tuple[float, str]] = field(default_factory=dict)  # outcome -> (cuota, casa)

    def best_odd(self, outcome: str) -> float | None:
        v = self.best.get(outcome)
        return v[0] if v else None


def parse_odds(raw: dict) -> dict[str, MarketOdds]:
    """Parsea la respuesta de /odds a {mercado: MarketOdds con la mejor cuota}. Puro."""
    out: dict[str, MarketOdds] = {}
    for item in raw.get("response", []):
        for bookmaker in item.get("bookmakers", []):
            book = bookmaker.get("name", "?")
            for bet in bookmaker.get("bets", []):
                market = bet.get("name", "")
                if not market:
                    continue
                mo = out.setdefault(market, MarketOdds(market=market))
                for value in bet.get("values", []):
                    outcome = value.get("value")
                    try:
                        odd = float(value.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if not outcome or odd <= 1.0:
                        continue
                    current = mo.best.get(outcome)
                    if current is None or odd > current[0]:
                        mo.best[outcome] = (odd, book)
    return out


def fetch_odds(key: str, fixture_id: int) -> dict[str, MarketOdds]:
    """Trae todas las cuotas de un partido (mejor cuota por resultado)."""
    raw = requests.get(
        f"{API_FOOTBALL_BASE}/odds", headers={"x-apisports-key": key},
        params={"fixture": fixture_id}, timeout=TIMEOUT_S,
    ).json()
    return parse_odds(raw)


def num_bookmakers(raw: dict) -> int:
    """Cuántas casas devolvió el partido (para saber la cobertura)."""
    resp = raw.get("response", [])
    return len(resp[0].get("bookmakers", [])) if resp else 0


def merge_odds(*sources: dict[str, MarketOdds]) -> dict[str, MarketOdds]:
    """Une varias fuentes de cuotas quedándose con la MEJOR (más alta) por resultado.

    Así sumamos casas de API-Football + odds-api.io y "leemos desde la primera hasta la
    última cuota": para cada mercado/resultado queda la que más paga.
    """
    out: dict[str, MarketOdds] = {}
    for src in sources:
        for market, mo in src.items():
            dest = out.setdefault(market, MarketOdds(market=market))
            for outcome, (odd, book) in mo.best.items():
                cur = dest.best.get(outcome)
                if cur is None or odd > cur[0]:
                    dest.best[outcome] = (odd, book)
    return out
