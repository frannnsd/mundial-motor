"""Métricas de evaluación probabilística para mercados 1X2.

- **RPS** (Ranked Probability Score): la métrica estándar para 1X2 porque respeta
  el orden (predecir empate cuando ganó el local es "menos malo" que predecir visita).
  0 = perfecto. Un buen modelo de selecciones ronda 0.17–0.19.
- **Brier** multiclase: error cuadrático medio de las probabilidades. Mide calibración.
- **Log-loss**: castiga fuerte la confianza equivocada. Bueno para optimizar.

Las tres: más bajo = mejor. Orden de resultados: [local, empate, visitante].
"""

from __future__ import annotations

import math

_EPS = 1e-15


def _onehot(actual_index: int) -> list[int]:
    e = [0, 0, 0]
    e[actual_index] = 1
    return e


def rps_1x2(probs: list[float], actual_index: int) -> float:
    """Ranked Probability Score de una predicción 1X2. 0 = perfecto."""
    e = _onehot(actual_index)
    cum_p = cum_e = 0.0
    total = 0.0
    for i in range(2):  # r - 1 = 2 cortes acumulados
        cum_p += probs[i]
        cum_e += e[i]
        total += (cum_p - cum_e) ** 2
    return total / 2.0


def brier_1x2(probs: list[float], actual_index: int) -> float:
    """Brier multiclase: suma de (pᵢ − oᵢ)². 0 = perfecto."""
    e = _onehot(actual_index)
    return sum((p - o) ** 2 for p, o in zip(probs, e, strict=True))


def log_loss_1x2(probs: list[float], actual_index: int) -> float:
    """Log-loss de la predicción: −log(prob del resultado real)."""
    p = max(probs[actual_index], _EPS)
    return -math.log(p)


def outcome_index(home_score: int, away_score: int) -> int:
    """Índice del resultado: 0 local gana, 1 empate, 2 visitante gana."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2
