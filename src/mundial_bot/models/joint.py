"""Combinadas del MISMO partido con probabilidad CONJUNTA (no multiplicar independiente).

Las patas de un mismo partido están correlacionadas: "gana el local" y "menos de 2.5
goles" no son independientes (si gana 3-1 cubre una pero no la otra). Para las patas de
goles (1X2, doble oportunidad, totales, ambos marcan, hándicap, total por equipo) la
probabilidad conjunta es EXACTA: se suma la matriz de marcadores en las celdas que
cumplen TODAS las condiciones. Córners y tarjetas vienen de otros modelos: se asumen
independientes de los goles y se multiplican (aproximación razonable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.count_market import over_under
from mundial_bot.models.goals_model import GoalsModel, GoalsModelError

_GOALS_MARKETS = {"ganador", "doble", "goles", "ambos_marcan", "handicap", "total_equipo"}
_COUNT_MARKETS = {"corners", "cards"}


@dataclass(frozen=True)
class JointResult:
    combined_prob: float
    fair_odds: float
    legs: list[tuple[str, float]]   # (descripción, prob marginal)
    note: str


def _require_half_line(line: float, market: str) -> None:
    """Las líneas enteras tienen push (devolución) que NO se modela en combinadas:
    se subestimaría la probabilidad. Exigimos línea .5 para no mentir."""
    if float(line).is_integer():
        raise ValueError(
            f"la línea {line:g} de '{market}' es entera (tiene push/devolución), que no se "
            "modela en combinadas. Usá una línea .5 (ej. 2.5 / -1.5)."
        )


def _goals_mask(leg: dict, i, j, margin, total):
    """Máscara booleana (sobre la matriz) de una pata de goles."""
    market = leg["market"]
    side = leg.get("side", "")
    line = float(leg.get("line", 0) or 0)
    team = leg.get("team", "home")
    if market == "ganador":
        return {"home": margin > 0, "draw": margin == 0, "away": margin < 0}[side]
    if market == "doble":
        return {"home_draw": margin >= 0, "home_away": margin != 0,
                "draw_away": margin <= 0}[side]
    if market == "goles":
        _require_half_line(line, "goles")
        return total > line if side == "over" else total < line
    if market == "ambos_marcan":
        yes = (i >= 1) & (j >= 1)
        return yes if side == "yes" else ~yes
    if market == "handicap":
        _require_half_line(line, "handicap")
        adj = (margin + line) if team == "home" else (-margin + line)
        return adj > 0
    if market == "total_equipo":
        _require_half_line(line, "total por equipo")
        g = i if team == "home" else j
        return g > line if side == "over" else g < line
    raise ValueError(f"mercado de goles desconocido: {market}")


def _count_prob(leg: dict, corners: CornersModel | None, cards: CardsModel | None,
                home: str, away: str) -> float:
    """Probabilidad marginal de una pata de córners/tarjetas."""
    market = leg["market"]
    side = leg.get("side", "over")
    line = float(leg.get("line", 0) or 0)
    if market == "corners" and corners is not None:
        pred = corners.predict(home, away)
        var = pred.total * corners.dispersion
        p_over, p_under = over_under(pred.total, line, variance=var)
    elif market == "cards" and cards is not None:
        pred = cards.predict(home, away)
        var = pred.total * getattr(cards, "dispersion", 1.0)
        p_over, p_under = over_under(pred.total, line, variance=var)
    else:
        raise ValueError(f"sin modelo para {market}")
    return p_over if side == "over" else p_under


def joint_same_match(
    home: str, away: str, *, goals: GoalsModel,
    corners: CornersModel | None, cards: CardsModel | None,
    legs: list[dict], neutral: bool = True,
) -> JointResult:
    """Probabilidad conjunta de una combinada de patas del MISMO partido."""
    if not goals.can_predict(home, away):
        raise GoalsModelError(f"sin datos de goles para {home} o {away}")
    matrix, _, _ = goals.score_matrix(home, away, neutral=neutral)
    n = matrix.shape[0]
    idx = np.arange(n)
    i = idx.reshape(-1, 1)
    j = idx.reshape(1, -1)
    margin = i - j
    total = i + j

    desc: list[tuple[str, float]] = []
    mask = np.ones((n, n), dtype=bool)
    count_prob = 1.0
    for leg in legs:
        market = leg.get("market", "")
        label = leg.get("desc") or f"{market} {leg.get('side','')} {leg.get('line','')}".strip()
        if market in _GOALS_MARKETS:
            m = _goals_mask(leg, i, j, margin, total)
            mask = mask & np.broadcast_to(m, (n, n))
            desc.append((label, float(matrix[np.broadcast_to(m, (n, n))].sum())))
        elif market in _COUNT_MARKETS:
            p = _count_prob(leg, corners, cards, home, away)
            count_prob *= p
            desc.append((label, p))
        else:
            raise ValueError(f"mercado desconocido en la pata: {market}")

    goals_joint = float(matrix[mask].sum())
    combined = goals_joint * count_prob
    fair = round(1.0 / combined, 2) if combined > 1e-9 else 0.0
    has_count = any(leg.get("market") in _COUNT_MARKETS for leg in legs)
    note = (
        "Patas de goles: probabilidad CONJUNTA exacta (correlación considerada)."
        + (" Córners/tarjetas: multiplicadas como independientes." if has_count else "")
    )
    return JointResult(combined_prob=combined, fair_odds=fair, legs=desc, note=note)
