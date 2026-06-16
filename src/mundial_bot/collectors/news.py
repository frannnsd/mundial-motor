"""Colector de noticias (NewsAPI) — contexto extra para el agente.

Trae titulares recientes de un equipo (suspensiones, lesiones de último momento,
motivación, clima, líos internos) para que Apu los pese en el análisis. Necesita
`NEWS_API_KEY` (newsapi.org). Sin key, devuelve un aviso claro.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
TIMEOUT_S = 20
DEFAULT_PAGE_SIZE = 6


@dataclass(frozen=True)
class Headline:
    title: str
    source: str
    published: str
    url: str
    description: str = ""


def parse_articles(raw: dict, *, limit: int = DEFAULT_PAGE_SIZE) -> list[Headline]:
    """Parsea la respuesta de NewsAPI v2 a titulares. Puro/defensivo."""
    out: list[Headline] = []
    for a in (raw.get("articles") or [])[:limit]:
        title = a.get("title")
        if not title:
            continue
        out.append(Headline(
            title=title,
            source=((a.get("source") or {}).get("name")) or "",
            published=(a.get("publishedAt") or "")[:10],
            url=a.get("url") or "",
            description=a.get("description") or "",
        ))
    return out


def fetch_team_news(
    key: str, team: str, *, language: str = "es", limit: int = DEFAULT_PAGE_SIZE,
) -> list[Headline]:
    """Titulares recientes de un equipo (los más nuevos primero)."""
    if not key:
        return []
    query = f'"{team}" (fútbol OR mundial OR selección OR lesión OR suspensión)'
    r = requests.get(
        NEWSAPI_BASE,
        params={
            "q": query, "language": language, "sortBy": "publishedAt",
            "pageSize": limit, "apiKey": key,
        },
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    return parse_articles(r.json(), limit=limit)


def format_news(team: str, headlines: list[Headline]) -> str:
    """Titulares para el agente (texto plano)."""
    if not headlines:
        return f"Sin noticias recientes de {team} (o sin NEWS_API_KEY)."
    lines = [f"NOTICIAS recientes de {team} (pesá vos qué importa):"]
    for h in headlines:
        src = f" — {h.source}" if h.source else ""
        when = f" ({h.published})" if h.published else ""
        lines.append(f"  • {h.title}{src}{when}")
    return "\n".join(lines)
