"""Colector de lesiones/suspensiones desde API-Football (/injuries).

Si a un equipo le faltan titulares clave, su fuerza baja. Acá traemos quién está
afuera (lesionado o suspendido) por partido o por equipo, mapeado al nombre del modelo.
El ajuste de fuerza a partir de esto vive en el modelo (penalización por bajas).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import requests

from mundial_bot.value.team_aliases import normalize_team

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
TIMEOUT_S = 20


@dataclass(frozen=True)
class Injury:
    team: str       # nombre normalizado (modelo)
    player: str
    reason: str     # "Injury", "Suspended", etc.
    kind: str       # type del API ("Missing Fixture", "Questionable", ...)


def parse_injuries(raw: dict) -> dict[str, list[Injury]]:
    """Agrupa las bajas por equipo (nombre normalizado). Puro."""
    out: dict[str, list[Injury]] = defaultdict(list)
    for item in raw.get("response", []):
        team = (item.get("team") or {}).get("name")
        player = (item.get("player") or {}).get("name")
        if not team or not player:
            continue
        p = item.get("player", {})
        out[normalize_team(team)].append(
            Injury(
                team=normalize_team(team),
                player=player,
                reason=p.get("reason", "") or "",
                kind=p.get("type", "") or "",
            )
        )
    return dict(out)


def fetch_injuries(
    key: str, *, fixture_id: int | None = None,
    league: int = WORLD_CUP_LEAGUE_ID, season: int = DEFAULT_SEASON,
) -> dict[str, list[Injury]]:
    """Trae las bajas de un partido (fixture_id) o de todo el torneo."""
    params: dict[str, int] = {}
    if fixture_id is not None:
        params["fixture"] = fixture_id
    else:
        params["league"] = league
        params["season"] = season
    raw = requests.get(
        f"{API_FOOTBALL_BASE}/injuries", headers={"x-apisports-key": key},
        params=params, timeout=TIMEOUT_S,
    ).json()
    return parse_injuries(raw)


def injury_counts(injuries: dict[str, list[Injury]]) -> dict[str, int]:
    """Cantidad de bajas por equipo (proxy simple de impacto)."""
    return {team: len(lst) for team, lst in injuries.items()}
