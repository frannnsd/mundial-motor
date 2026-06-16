"""Blend del 1X2: combina Elo (mejor baseline de ganador en data rala) con la matriz
Dixon-Coles (estructura de goles del partido).

Los dos suelen diferir (ej. Argentina: Elo 78% vs DC 52%). En vez de que el agente los
reconcilie a mano, los fusionamos en UNA probabilidad coherente: ensemble ponderado.

Hallazgo del backtest (scripts/backtest_blend.py, 2696 partidos out-of-sample): al revés
de lo que se asumía, **Dixon-Coles le gana a Elo en 1X2** (RPS 0.173 vs 0.177), y el
blend le gana a los dos (0.1715). El óptimo está en w_elo≈0.30-0.50 (DC manda). Usamos
0.40: dentro del óptimo y un toque más de Elo por robustez en cruces raros del Mundial
(poca data, donde DC se puede desestabilizar). `W_ELO` es recalibrable con el backtest.
"""

from __future__ import annotations

# Peso de Elo en el blend de 1X2 (el resto va a Dixon-Coles). Validado out-of-sample.
W_ELO = 0.40


def blend_1x2(
    elo: tuple[float, float, float],
    dc: tuple[float, float, float],
    *,
    w_elo: float = W_ELO,
) -> tuple[float, float, float]:
    """Combina dos triples (home, draw, away) en uno normalizado. w_elo ∈ [0,1]."""
    w = min(max(w_elo, 0.0), 1.0)
    home = w * elo[0] + (1 - w) * dc[0]
    draw = w * elo[1] + (1 - w) * dc[1]
    away = w * elo[2] + (1 - w) * dc[2]
    total = home + draw + away
    if total <= 0:
        return elo
    return (home / total, draw / total, away / total)
