"""Distribuciones de conteo + métricas (CRPS, calibración) — la vara ÚNICA.

Todos los cerebros expresan su predicción como (media, varianza) y este módulo
las convierte en una pmf discreta con la MISMA regla (NegBin si hay sobre-
dispersión, Poisson si no). Así la competencia compara métodos de estimación,
no familias de distribución distintas.

CRPS discreto (Ranked Probability Score generalizado a conteos):
    CRPS(F, y) = Σ_k (F(k) − 1{y ≤ k})²
Premia media Y dispersión correctas; más bajo = mejor. Determinístico (sin RNG).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

# Truncamiento de la pmf por cantidad (cubre >99.9% de la masa observada).
GRID_MAX: dict[str, int] = {
    "goals": 12, "corners": 25, "yellows": 14, "shots": 45, "sot": 22,
}
DEFAULT_GRID = 40


def quantity_grid(quantity: str) -> int:
    """Tamaño de grilla para la familia de la cantidad (goals_h → goals)."""
    family = quantity.rsplit("_", 1)[0]
    return GRID_MAX.get(family, DEFAULT_GRID)


def count_pmf(mean: float, variance: float, k_max: int) -> np.ndarray:
    """pmf discreta en [0, k_max] desde (media, varianza) — regla común a todos.

    NegBin (momentos: r = m²/(v−m), p = r/(r+m)) si v > m; Poisson si no.
    La masa de la cola (> k_max) se acumula en el último bin para que sume 1.
    """
    m = max(float(mean), 1e-6)
    v = float(variance)
    k = np.arange(k_max + 1)
    if v > m + 1e-9:
        r = m * m / (v - m)
        p = r / (r + m)
        pmf = nbinom.pmf(k, r, p)
    else:
        pmf = poisson.pmf(k, m)
    pmf = np.asarray(pmf, dtype=float)
    tail = max(0.0, 1.0 - float(pmf.sum()))
    pmf[-1] += tail
    return pmf


def crps_count(pmf: np.ndarray, actual: int) -> float:
    """CRPS discreto de una pmf contra el valor observado (más bajo = mejor)."""
    cdf = np.cumsum(pmf)
    y = min(int(actual), len(pmf) - 1)
    step = (np.arange(len(pmf)) >= y).astype(float)
    return float(np.sum((cdf - step) ** 2))


def convolve_pmf(pmf_a: np.ndarray, pmf_b: np.ndarray, k_max: int | None = None) -> np.ndarray:
    """pmf de la SUMA de dos conteos independientes (local + visita → total).

    Caveat honesto: asume independencia local/visita (sin correlación tipo
    Dixon-Coles tau). Para totales de córners/remates/tarjetas el efecto es
    menor; para goles es una aproximación conocida (flagged en el reporte).
    """
    total = np.convolve(pmf_a, pmf_b)
    if k_max is not None and len(total) > k_max + 1:
        head = total[: k_max + 1].copy()
        head[-1] += float(total[k_max + 1 :].sum())
        total = head
    s = float(total.sum())
    return total / s if s > 0 else total


def p_over(pmf: np.ndarray, line: float) -> float:
    """P(conteo > línea .5) desde la pmf (línea 9.5 → P(X ≥ 10))."""
    thresh = int(np.floor(line)) + 1
    if thresh >= len(pmf):
        return 0.0
    return float(pmf[thresh:].sum())


def calibration_table(
    pairs: list[tuple[float, bool]], bins: int = 10
) -> tuple[list[dict], float]:
    """Tabla de fiabilidad + ECE (error de calibración esperado, ponderado por n).

    pairs: [(prob_predicha, ocurrió)] — ej. P(over 9.5) vs si hubo over.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    n_total = len(pairs)
    ece = 0.0
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sub = [(p, o) for p, o in pairs if p >= lo and (p < hi or i == bins - 1)]
        if not sub:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "pred": None, "real": None})
            continue
        pred = float(np.mean([p for p, _ in sub]))
        real = float(np.mean([float(o) for _, o in sub]))
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sub),
                     "pred": round(pred, 3), "real": round(real, 3)})
        ece += (len(sub) / n_total) * abs(pred - real)
    return rows, float(ece)
