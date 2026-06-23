"""Tiros por JUGADOR (player props) desde API-Football /players.

Trae los tiros (totales y al arco) por partido de un jugador y calcula la probabilidad
de over/under por línea (0.5 / 1.5 / 2.5...) con Poisson. Para "¿cuántos tiros mete
Messi?" y para sumar patas de jugador a las combinadas.

Caveats honestos: asume que el jugador ARRANCA de titular (las alineaciones salen ~1h
antes). La tasa sale de su temporada (club + selección). No ajusta por rival.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import requests
from scipy.stats import poisson

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
TIMEOUT_S = 25
SOT_LINES = (0.5, 1.5, 2.5)
SHOT_LINES = (0.5, 1.5, 2.5, 3.5)


@dataclass(frozen=True)
class PlayerShots:
    name: str
    team: str
    appearances: int
    shots_total: int
    shots_on: int

    @property
    def shots_per_game(self) -> float:
        return self.shots_total / self.appearances if self.appearances else 0.0

    @property
    def sot_per_game(self) -> float:
        return self.shots_on / self.appearances if self.appearances else 0.0


def _strip(s: str) -> str:
    norm = unicodedata.normalize("NFD", (s or "").lower().strip())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def _num(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def parse_player_shots(raw: dict, query: str) -> PlayerShots | None:
    """Suma tiros/tiros-al-arco del jugador a lo largo de su temporada (club + selección)."""
    response = raw.get("response", [])
    if not response:
        return None
    q = _strip(query)
    # Elegir el item cuyo nombre matchea mejor la búsqueda.
    best = None
    for item in response:
        name = (item.get("player") or {}).get("name", "")
        if q in _strip(name) or _strip(name) in q:
            best = item
            break
    best = best or response[0]
    name = (best.get("player") or {}).get("name", query)

    appearances = shots_total = shots_on = 0
    team = ""
    for st in best.get("statistics", []) or []:
        appearances += _num((st.get("games") or {}).get("appearences"))
        shots = st.get("shots") or {}
        shots_total += _num(shots.get("total"))
        shots_on += _num(shots.get("on"))
        if not team:
            team = (st.get("team") or {}).get("name", "") or ""
    if appearances <= 0:
        return None
    return PlayerShots(name=name, team=team, appearances=appearances,
                       shots_total=shots_total, shots_on=shots_on)


def _get(key: str, params: dict) -> dict:
    r = requests.get(
        f"{API_FOOTBALL_BASE}/players",
        headers={"x-apisports-key": key}, params=params, timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def _find_player_id(raw: dict, query: str) -> int | None:
    """Elige el id del jugador que mejor matchea el nombre buscado."""
    q = _strip(query)
    items = raw.get("response", [])
    for item in items:
        name = (item.get("player") or {}).get("name", "")
        if q in _strip(name) or _strip(name) in q:
            return (item.get("player") or {}).get("id")
    return ((items[0].get("player") or {}).get("id")) if items else None


def fetch_player_shots(
    key: str, name: str, *, season: int = DEFAULT_SEASON
) -> PlayerShots | None:
    """Tiros de un jugador del Mundial en la temporada (club + selección). None si no hay.

    La API exige league/team con search → buscamos en el Mundial (league=1) para resolver
    el id, y después traemos TODAS sus stats de la temporada por id (muestra más rica).
    """
    if not key or len(name.strip()) < 3:
        return None
    search = _get(key, {"search": name.strip(), "league": WORLD_CUP_LEAGUE_ID, "season": season})
    pid = _find_player_id(search, name)
    if pid is None:
        return None
    full = _get(key, {"id": pid, "season": season})
    # Preferimos las stats completas por id; si vinieran vacías, usamos las del Mundial.
    return parse_player_shots(full, name) or parse_player_shots(search, name)


def _poisson_over_under(mean: float, lines: tuple[float, ...]) -> list[tuple[float, float]]:
    """[(línea, prob_over)] por Poisson(mean)."""
    out = []
    for line in lines:
        k = int(line)  # over line .5 → más de k
        p_over = float(1.0 - poisson.cdf(k, mean)) if mean > 0 else 0.0
        out.append((line, p_over))
    return out


def opponent_factor(shots_model, rival: str, *, lo: float = 0.6, hi: float = 1.6) -> float:
    """Factor defensivo del rival: tiros al arco que concede vs la media (1.0 = media).

    >1 = defensa floja (el jugador patea MÁS); <1 = defensa firme (patea menos).
    Acotado para no exagerar con muestras chicas.
    """
    avg = getattr(shots_model, "league_avg", 0.0)
    conceded = getattr(shots_model, "team_against", {}).get(rival, avg)
    if avg <= 0:
        return 1.0
    return min(max(conceded / avg, lo), hi)


def format_player_shots(
    ps: PlayerShots, *, opponent: str | None = None, factor: float | None = None
) -> str:
    """Texto para el agente: tasa + over/under. Si hay `factor`, ajusta por el rival."""
    sot_rate, shot_rate = ps.sot_per_game, ps.shots_per_game
    lines = [
        f"TIROS DE {ps.name} ({ps.team}) — {ps.appearances} partidos esta temporada:",
        f"  base: {ps.shots_per_game:.2f} tiros · {ps.sot_per_game:.2f} al arco por partido",
    ]
    if factor and opponent:
        sot_rate, shot_rate = ps.sot_per_game * factor, ps.shots_per_game * factor
        tag = "defensa floja" if factor > 1.05 else "defensa firme" if factor < 0.95 else "neutra"
        lines.append(
            f"  ajustado vs {opponent} ({tag}, concede {factor:.2f}x la media): "
            f"{shot_rate:.2f} tiros · {sot_rate:.2f} al arco"
        )
    lines.append("  (asume que arranca de titular)")
    lines.append("  Tiros al arco — probabilidad de que pase:")
    for line, p in _poisson_over_under(sot_rate, SOT_LINES):
        lines.append(f"    Más de {line:g}: {p:.0%}")
    lines.append("  Tiros totales:")
    for line, p in _poisson_over_under(shot_rate, SHOT_LINES):
        lines.append(f"    Más de {line:g}: {p:.0%}")
    return "\n".join(lines)
