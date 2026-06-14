"""Utilidades para mercados de conteo (córners, tarjetas) con distribución de Poisson.

Dado un valor esperado (ej. 10.3 córners), calculamos la probabilidad de over/under
de cualquier línea. Poisson es el baseline estándar; para córners/tarjetas hay sobre-
dispersión y la Negative Binomial afina mejor — queda como mejora futura.
"""

from __future__ import annotations

import math

from scipy.stats import nbinom, poisson

# Líneas típicas que ofrecen las casas.
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)

# Fuerza de regularización: equivale a "K partidos de prior" hacia la media.
SHRINKAGE_K = 5.0


def shrink(value: float, count: float, prior: float, k: float = SHRINKAGE_K) -> float:
    """Regulariza una tasa hacia un prior según el tamaño de muestra.

    Con pocos partidos (count bajo), tira fuerte hacia la media de la liga; con
    muchos, confía en la tasa del equipo. Evita predicciones extremas por muestra chica.
    """
    return (count * value + k * prior) / (count + k)


def _nb_params(mean: float, variance: float) -> tuple[float, float] | None:
    """Parámetros (r, p) de la Negative Binomial por método de momentos.

    var = mean + mean²/r → r = mean²/(var−mean), p = r/(r+mean).
    Devuelve None si no hay sobre-dispersión (var ≤ mean) → usar Poisson.
    """
    if variance <= mean:
        return None
    r = mean**2 / (variance - mean)
    p = r / (r + mean)
    return r, p


def over_under(
    expected: float, line: float, *, variance: float | None = None
) -> tuple[float, float]:
    """Probabilidad de (over, under) de una línea.

    Usa Negative Binomial si se pasa ``variance`` y hay sobre-dispersión (mejor para
    córners/tarjetas); si no, cae a Poisson(expected).
    """
    if expected <= 0:
        return 0.0, 1.0
    k = math.floor(line)
    params = _nb_params(expected, variance) if variance is not None else None
    if params is not None:
        r, p = params
        p_over = float(1.0 - nbinom.cdf(k, r, p))
    else:
        p_over = float(1.0 - poisson.cdf(k, expected))
    return p_over, 1.0 - p_over


def closest_line(expected: float, lines: tuple[float, ...]) -> float:
    """La línea .5 más cercana al valor esperado (la más 'pareja')."""
    return min(lines, key=lambda line: abs(line - expected))


# Confianza objetivo para elegir la línea: firme pero no trivial.
_CONF_CAP = 0.78


def best_line(expected: float, lines: tuple[float, ...], *, variance: float | None = None) -> float:
    """Elige la línea con la predicción más FIRME (no la más pareja).

    Maximiza la confianza del lado favorito sin pasar de ~0.78 (evita líneas
    triviales tipo 99%). Así el bot apuesta donde tiene convicción, no en el ~50/50.
    """
    best, best_score = lines[0], -1.0
    for line in lines:
        p_over, p_under = over_under(expected, line, variance=variance)
        conf = max(p_over, p_under)
        score = conf if conf <= _CONF_CAP else (2 * _CONF_CAP - conf)
        if score > best_score:
            best, best_score = line, score
    return best
