"""Gestión de banca con criterio de Kelly fraccionado — Agente 4.

Kelly óptimo: `f* = (p·cuota − 1) / (cuota − 1)` = edge / (cuota−1). Apuesta
proporcional al edge.

Usamos **¼ Kelly** (no full Kelly) porque nuestras probabilidades de selecciones
son ruidosas: full Kelly sobre probabilidades imperfectas sobre-apuesta y genera
drawdowns brutales. Además topamos cada apuesta (3% de la banca) y la exposición
simultánea total (25%), reescalando si hace falta.
"""

from __future__ import annotations

from dataclasses import dataclass

from mundial_bot.value.ev import ValuePick


def kelly_fraction(prob: float, odds: float) -> float:
    """Fracción de Kelly completa para una apuesta. 0 si no hay edge."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * odds - 1.0) / b
    return max(0.0, f)


@dataclass(frozen=True)
class StakeConfig:
    bankroll: float = 100.0
    kelly_fraction: float = 0.25       # ¼ Kelly
    max_stake_pct: float = 0.03        # tope por apuesta: 3% de la banca
    max_total_exposure_pct: float = 0.25  # tope de exposición total: 25%


@dataclass(frozen=True)
class StakedPick:
    """Un value pick con su stake calculado."""

    pick: ValuePick
    stake: float            # monto a apostar (USD)
    full_kelly: float       # fracción de Kelly completa (antes de ¼ y topes)
    applied_fraction: float # fracción de banca finalmente aplicada


def _raw_fraction(pick: ValuePick, config: StakeConfig) -> tuple[float, float]:
    """Fracción de banca antes de reescalar por exposición total. (full_kelly, frac)."""
    fk = kelly_fraction(pick.model_prob, pick.selection.odds)
    frac = min(fk * config.kelly_fraction, config.max_stake_pct)
    return fk, frac


def stake_for(pick: ValuePick, config: StakeConfig) -> float:
    """Stake en USD para un único pick (¼ Kelly con tope por apuesta)."""
    _, frac = _raw_fraction(pick, config)
    return round(frac * config.bankroll, 2)


def size_portfolio(picks: list[ValuePick], config: StakeConfig) -> list[StakedPick]:
    """Calcula los stakes de una cartera, respetando el tope de exposición total.

    Si la suma de stakes supera ``max_total_exposure_pct`` de la banca, se reescalan
    todos proporcionalmente.
    """
    fractions = [_raw_fraction(p, config) for p in picks]
    total_frac = sum(frac for _, frac in fractions)

    scale = 1.0
    if total_frac > config.max_total_exposure_pct and total_frac > 0:
        scale = config.max_total_exposure_pct / total_frac

    staked: list[StakedPick] = []
    for pick, (fk, frac) in zip(picks, fractions, strict=True):
        applied = frac * scale
        staked.append(
            StakedPick(
                pick=pick,
                stake=round(applied * config.bankroll, 2),
                full_kelly=fk,
                applied_fraction=applied,
            )
        )
    return staked
