"""Lector de cuotas de odds-api.io (https://odds-api.io) — fuente extra de casas.

Suma cuotas a las de API-Football: para cada mercado/resultado el evaluador después se
queda con la que más paga (ver `merge_odds`). Así "leemos desde la primera hasta la
última cuota".

API (https://api.odds-api.io/v3, auth por query `apiKey`):
  - GET /leagues?sport=football            → ligas (el Mundial es `international-fifa-world-cup`)
  - GET /events?sport=football&league=..   → partidos (id, equipos, fecha, status, scores)
  - GET /odds?eventId=..&bookmakers=A,B    → cuotas (markets: ML=1X2, Totals=goles, Spread=hándicap)

⚠️ El plan de Franco permite máx. 2 casas por request (`bookmakers`), por nombre exacto
(ver GET /bookmakers). Por eso usamos 2 casas grandes con cuotas generosas.
"""

from __future__ import annotations

import unicodedata

import requests

from mundial_bot.collectors.odds_af import MarketOdds

ODDSAPI_BASE = "https://api.odds-api.io/v3"
WORLD_CUP_LEAGUE = "international-fifa-world-cup"
# Máx. 2 por el plan; alineado a las casas de Franco (Bet365 + Betano).
DEFAULT_BOOKMAKERS = "Bet365,Betano"
TIMEOUT_S = 25

# Nombres de mercado de odds-api.io → los que consume el evaluador (estilo API-Football).
_ML_OUTCOME = {"home": "Home", "draw": "Draw", "away": "Away"}


def _norm(name: str) -> str:
    """Normaliza un nombre de equipo (sin acentos, minúsculas) para cruzar fuentes."""
    norm = unicodedata.normalize("NFD", (name or "").lower().strip())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def _fmt_line(hdp) -> str:
    """Formatea la línea de goles igual que el modelo (2.5, 1.5...) para que crucen."""
    try:
        return f"{float(hdp):g}"
    except (TypeError, ValueError):
        return str(hdp)


def fetch_wc_events(key: str, *, league: str = WORLD_CUP_LEAGUE) -> list[dict]:
    """Trae los partidos del Mundial (jugados + en vivo + por jugar)."""
    r = requests.get(
        f"{ODDSAPI_BASE}/events",
        params={"sport": "football", "league": league, "apiKey": key},
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("data", [])


def find_event_id(events: list[dict], home: str, away: str) -> int | None:
    """Busca el id del partido por nombres de equipo (sin importar local/visita)."""
    target = frozenset({_norm(home), _norm(away)})
    for ev in events:
        pair = frozenset({_norm(ev.get("home", "")), _norm(ev.get("away", ""))})
        if pair == target:
            return ev.get("id")
    return None


def fetch_event_odds(
    key: str, event_id: int, *, bookmakers: str = DEFAULT_BOOKMAKERS
) -> dict:
    """Trae las cuotas crudas de un partido (máx. 2 casas por el plan)."""
    r = requests.get(
        f"{ODDSAPI_BASE}/odds",
        params={"eventId": event_id, "bookmakers": bookmakers, "apiKey": key},
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def parse_event_odds(raw: dict) -> dict[str, MarketOdds]:
    """Convierte la respuesta de /odds a {mercado: MarketOdds con la mejor cuota}.

    Mapea ML→'Match Winner' (Home/Draw/Away) y Totals→'Goals Over/Under' (Over/Under X.X),
    quedándose con la cuota más alta entre las casas pedidas.
    """
    out: dict[str, MarketOdds] = {}
    books = raw.get("bookmakers", {}) or {}
    for book, markets in books.items():
        for market in markets or []:
            name = market.get("name", "")
            rows = market.get("odds", []) or []
            if name == "ML":
                mo = out.setdefault("Match Winner", MarketOdds(market="Match Winner"))
                for row in rows:
                    for raw_key, outcome in _ML_OUTCOME.items():
                        _put(mo, outcome, row.get(raw_key), book)
            elif name == "Totals":
                mo = out.setdefault("Goals Over/Under", MarketOdds(market="Goals Over/Under"))
                for row in rows:
                    line = _fmt_line(row.get("hdp"))
                    _put(mo, f"Over {line}", row.get("over"), book)
                    _put(mo, f"Under {line}", row.get("under"), book)
    return out


def _put(mo: MarketOdds, outcome: str, raw_odd, book: str) -> None:
    """Guarda la cuota si es válida y mejora la mejor actual."""
    try:
        odd = float(raw_odd)
    except (TypeError, ValueError):
        return
    if odd <= 1.0:
        return
    cur = mo.best.get(outcome)
    if cur is None or odd > cur[0]:
        mo.best[outcome] = (odd, book)


def fetch_match_odds(
    key: str, home: str, away: str, *,
    events: list[dict] | None = None, bookmakers: str = DEFAULT_BOOKMAKERS,
) -> dict[str, MarketOdds]:
    """Cuotas de odds-api.io para un partido (cruza por nombres). {} si no lo encuentra."""
    if not key:
        return {}
    if events is None:
        events = fetch_wc_events(key)
    event_id = find_event_id(events, home, away)
    if event_id is None:
        return {}
    return parse_event_odds(fetch_event_odds(key, event_id, bookmakers=bookmakers))
