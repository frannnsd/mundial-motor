"""Utilidades para mercados de conteo (córners, tarjetas) con distribución de Poisson.

Dado un valor esperado (ej. 10.3 córners), calculamos la probabilidad de over/under
de cualquier línea. Poisson es el baseline estándar; para córners/tarjetas hay sobre-
dispersión y la Negative Binomial afina mejor — queda como mejora futura.
"""

from __future__ import annotations

import math

from scipy.stats import poisson

# Líneas típicas que ofrecen las casas.
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


def over_under(expected: float, line: float) -> tuple[float, float]:
    """Probabilidad de (over, under) de una línea, asumiendo Poisson(expected)."""
    k = math.floor(line)
    p_over = float(1.0 - poisson.cdf(k, expected))
    return p_over, 1.0 - p_over


def closest_line(expected: float, lines: tuple[float, ...]) -> float:
    """La línea .5 más cercana al valor esperado (la más 'pareja')."""
    return min(lines, key=lambda line: abs(line - expected))
