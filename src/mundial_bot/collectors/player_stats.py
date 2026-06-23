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
_SQUAD_MIN_APPS = 2     # ignora jugadores con muy pocos partidos
_SQUAD_MAX_PAGES = 4


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


@dataclass(frozen=True)
class SquadGoals:
    name: str
    appearances: int
    goals: int
    shots_on: int = 0

    @property
    def goals_per_game(self) -> float:
        return self.goals / self.appearances if self.appearances else 0.0

    @property
    def sot_per_game(self) -> float:
        return self.shots_on / self.appearances if self.appearances else 0.0


def team_id_map(key: str, *, season: int = DEFAULT_SEASON) -> dict[str, int]:
    """{nombre_normalizado_del_modelo: team_id} de los equipos del Mundial."""
    from mundial_bot.value.team_aliases import normalize_team

    r = requests.get(
        f"{API_FOOTBALL_BASE}/teams",
        headers={"x-apisports-key": key},
        params={"league": WORLD_CUP_LEAGUE_ID, "season": season}, timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    out: dict[str, int] = {}
    for item in r.json().get("response", []):
        team = item.get("team") or {}
        if team.get("id") and team.get("name"):
            out[normalize_team(team["name"])] = int(team["id"])
    return out


def fetch_squad_goals(
    key: str, team_id: int, *, season: int = DEFAULT_SEASON
) -> list[SquadGoals]:
    """Plantel de un equipo con sus goles/partidos de la temporada (paginado)."""
    players: list[SquadGoals] = []
    page = 1
    while page <= _SQUAD_MAX_PAGES:
        raw = _get(key, {"team": team_id, "season": season, "page": page})
        for item in raw.get("response", []):
            name = (item.get("player") or {}).get("name", "")
            apps = goals = sot = 0
            for st in item.get("statistics", []) or []:
                apps += _num((st.get("games") or {}).get("appearences"))
                goals += _num((st.get("goals") or {}).get("total"))
                sot += _num((st.get("shots") or {}).get("on"))
            if name and apps > 0:
                players.append(SquadGoals(name=name, appearances=apps, goals=goals, shots_on=sot))
        paging = raw.get("paging") or {}
        if page >= _num(paging.get("total")):
            break
        page += 1
    return players


def goalscorer_probs(
    squad: list[SquadGoals], team_xg: float, *, min_apps: int = _SQUAD_MIN_APPS, top: int = 8
) -> list[tuple[str, float, float, float, float]]:
    """Reparte el xG del equipo (ya ajustado por el rival) entre los jugadores según su
    tasa de gol. Devuelve [(nombre, xg_jugador, P(1+), P(2+), P(3+))] ordenado por P(1+)."""
    elig = [p for p in squad if p.appearances >= min_apps and p.goals_per_game > 0]
    total_gpg = sum(p.goals_per_game for p in elig)
    if total_gpg <= 0 or team_xg <= 0:
        return []
    rows = []
    for p in elig:
        pxg = team_xg * (p.goals_per_game / total_gpg)
        p1 = float(1.0 - poisson.pmf(0, pxg))
        p2 = float(1.0 - poisson.cdf(1, pxg))
        p3 = float(1.0 - poisson.cdf(2, pxg))
        rows.append((p.name, pxg, p1, p2, p3))
    rows.sort(key=lambda r: -r[2])
    return rows[:top]


def player_sot_probs(
    squad: list[SquadGoals], factor: float = 1.0, *, min_apps: int = _SQUAD_MIN_APPS,
    top: int = 14,
) -> list[tuple[str, float, float]]:
    """P(1+ tiro al arco) por jugador = Poisson(su tasa de tiros al arco × factor del rival).

    Devuelve [(nombre, tasa_ajustada, P(1+))] ordenado por P(1+)."""
    rows = []
    for p in squad:
        if p.appearances < min_apps or p.sot_per_game <= 0:
            continue
        rate = p.sot_per_game * factor
        p1 = float(1.0 - poisson.pmf(0, rate))
        rows.append((p.name, rate, p1))
    rows.sort(key=lambda r: -r[2])
    return rows[:top]


def player_sot_casa_odds(best: dict) -> dict[str, tuple[float, str]]:
    """De las cuotas de 'Player Shots On Target' arma {nombre_normalizado: (cuota, casa)}.

    Los outcomes vienen como 'Lionel Messi - 1+'; nos quedamos con los de 1+.
    """
    out: dict[str, tuple[float, str]] = {}
    for outcome, val in best.items():
        if "1+" not in outcome:
            continue
        name = outcome.rsplit(" - ", 1)[0] if " - " in outcome else outcome
        out[_strip(name)] = val
    return out


def match_casa_odd(player_name: str, casa: dict[str, tuple[float, str]]):
    """Busca la cuota de un jugador por nombre (exacto o por apellido)."""
    norm = _strip(player_name)
    if norm in casa:
        return casa[norm]
    last = norm.split()[-1] if norm.split() else norm
    for cname, val in casa.items():
        if last and last == (cname.split()[-1] if cname.split() else cname):
            return val
    return None


def format_scorers(team: str, scorers: list[tuple[str, float, float, float, float]]) -> str:
    """Goleadores probables de un equipo (1+/2+/3+ goles)."""
    if not scorers:
        return f"⚽ {team}: sin datos de goleadores."
    lines = [f"⚽ <b>{team}</b> — chance de hacer gol (ya ajustado por el rival):"]
    for name, _pxg, p1, p2, p3 in scorers:
        extra = ""
        if p2 >= 0.04:
            extra += f" · 2+: {p2:.0%}"
        if p3 >= 0.015:
            extra += f" · 3+: {p3:.0%}"
        lines.append(f"   {name}: <b>1+ {p1:.0%}</b>{extra}")
    return "\n".join(lines)


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
