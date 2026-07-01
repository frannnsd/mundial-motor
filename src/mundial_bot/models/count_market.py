"""Utilidades para mercados de conteo (córners, tarjetas) con distribución de Poisson.

Dado un valor esperado (ej. 10.3 córners), calculamos la probabilidad de over/under
de cualquier línea. Poisson es el baseline estándar; para córners/tarjetas hay sobre-
dispersión y la Negative Binomial afina mejor — queda como mejora futura.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

# Líneas típicas que ofrecen las casas.
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)

# Fuerza de regularización: equivale a "K partidos de prior" hacia la media.
SHRINKAGE_K = 5.0


DEFAULT_HALFLIFE_DAYS = 200.0


def weighted_means(
    events: pd.DataFrame,
    cols: list[str],
    *,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    as_of: pd.Timestamp | str | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Media ponderada por recencia de varias columnas, por equipo.

    Los partidos recientes pesan más (decaimiento exponencial por fecha). Devuelve
    ({col: {team: media}}, {team: muestra_efectiva}). Si no hay fecha, peso uniforme.

    ``as_of`` (POINT-IN-TIME): si se pasa la fecha de kickoff, se EXCLUYE cualquier
    partido con fecha >= kickoff y el decaimiento se ancla al kickoff (no al último
    partido del cache). Con ``as_of=None`` el comportamiento es idéntico al histórico
    (ref = último partido del set) — así el path live de scoreo no cambia.
    """
    ev = events.copy()
    if "date" in ev.columns:
        dates = pd.to_datetime(ev["date"], errors="coerce")
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            # Decisión: `NaT < ts` es False, así que las filas con fecha NULA quedan
            # EXCLUIDAS bajo point-in-time (conservador: si no sabemos cuándo pasó, no
            # la usamos para no arriesgar leakage). En modo live (as_of=None) no se filtra.
            keep = dates < as_of_ts  # nunca datos del kickoff en adelante
            ev = ev.loc[keep].copy()
            dates = dates.loc[keep]
            ref = as_of_ts
        else:
            ref = dates.max() if dates.notna().any() else None
        if ref is not None and len(ev):
            days = (ref - dates).dt.days.clip(lower=0)
            days = days.fillna(days.max() if days.notna().any() else 0.0)
            ev["_w"] = np.exp(-np.log(2) / halflife_days * days)
        else:
            ev["_w"] = 1.0
    else:
        ev["_w"] = 1.0

    eff_count = ev.groupby("team")["_w"].sum()
    out: dict[str, dict[str, float]] = {}
    for col in cols:
        num = ev.assign(_vw=ev[col] * ev["_w"]).groupby("team")["_vw"].sum()
        out[col] = (num / eff_count).to_dict()
    return out, eff_count.to_dict()


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
