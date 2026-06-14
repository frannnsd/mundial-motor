"""Combinadas (parlays) — Agente 4.

Mito: "las combinadas siempre son -EV". Verdad: lo son cuando combinás apuestas
*justas* (la casa multiplica su margen en cada pata). PERO si **cada pata es +EV
según nuestro modelo** (p·cuota > 1) y las patas son **independientes** (partidos
distintos), el EV combinado también es positivo: `EV = Π(pᵢ·cuotaᵢ) − 1 > 0`.

El costo: la varianza se dispara. Por eso las combinadas se apuestan con stakes
chicos (Kelly sobre la prob/cuota combinada da un número pequeño) y siempre
mostramos el EV real. Ofrecemos dos sabores:
  - **conservadora**: pocas patas, mayor probabilidad combinada.
  - **alto riesgo**: más patas / mayor cuota combinada (más pago, más varianza).

⚠️ Asumimos independencia entre partidos distintos. Patas del MISMO partido están
correlacionadas y NO se combinan acá (se marcan `correlated=True` y Kelly = 0).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import prod

from mundial_bot.staking.kelly import kelly_fraction as _single_kelly
from mundial_bot.value.ev import ValuePick


@dataclass(frozen=True)
class Parlay:
    """Una combinada de varios value picks."""

    legs: tuple[ValuePick, ...]
    correlated: bool = False

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def combined_prob(self) -> float:
        """Probabilidad combinada (asume independencia entre patas)."""
        return prod(leg.model_prob for leg in self.legs)

    @property
    def combined_odds(self) -> float:
        return prod(leg.selection.odds for leg in self.legs)

    @property
    def combined_ev(self) -> float:
        """EV combinado por unidad: Π(pᵢ·cuotaᵢ) − 1."""
        return self.combined_prob * self.combined_odds - 1.0

    @property
    def kelly_fraction(self) -> float:
        """Kelly sobre la combinada. 0 si está correlacionada (no fiable)."""
        if self.correlated:
            return 0.0
        return _single_kelly(self.combined_prob, self.combined_odds)


def build_parlay(picks: list[ValuePick], *, correlated: bool = False) -> Parlay:
    return Parlay(legs=tuple(picks), correlated=correlated)


def suggest_parlays(
    picks: list[ValuePick],
    *,
    sizes: tuple[int, ...] = (2, 3),
    min_combined_ev: float = 0.0,
    max_results: int = 5,
) -> list[Parlay]:
    """Genera combinadas de value picks de partidos distintos, con EV combinado positivo.

    Solo combina patas de partidos distintos (independencia). Ordena por EV combinado.
    """
    out: list[Parlay] = []
    for size in sizes:
        for combo in itertools.combinations(picks, size):
            matches = {p.selection.match for p in combo}
            if len(matches) < size:
                continue  # hay patas del mismo partido → correlacionadas, se saltean
            par = Parlay(legs=tuple(combo))
            if par.combined_ev >= min_combined_ev:
                out.append(par)
    out.sort(key=lambda p: p.combined_ev, reverse=True)
    return out[:max_results]


def safest_parlay(parlays: list[Parlay]) -> Parlay | None:
    """La combinada con mayor probabilidad combinada (la más 'segura')."""
    return max(parlays, key=lambda p: p.combined_prob) if parlays else None


def highest_payout_parlay(parlays: list[Parlay]) -> Parlay | None:
    """La combinada de mayor cuota combinada (alto riesgo / alto pago)."""
    return max(parlays, key=lambda p: p.combined_odds) if parlays else None
