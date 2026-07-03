"""Proyección de mercados MLB desde las pmfs del cerebro unificado (M2).

Capa determinística común (no compite): traduce las distribuciones de carreras,
hits y carreras-F5 a los mercados de bet365. Diferencias clave vs fútbol:
- NO hay empate: el 9-innings igualado va a entradas extra. La masa del empate se
  reparte 50/50 (moneda documentada, no tuneada) → moneyline.
- F5 es EXACTO: viene de su propia familia (runs_f5), no de un escalado.
- Run line estándar ±1.5 con la matriz de márgenes.

Caveat honesto: independencia local/visita en la matriz (igual que fútbol);
en béisbol la correlación entre ofensivas es menor que en fútbol (no comparten
pelota) — sesgo chico, documentado.
"""

from __future__ import annotations

import numpy as np

from mundial_bot.research.distributions import convolve_pmf, p_over

EXTRA_INNINGS_HOME_P = 0.5  # empate a 9 → moneda (documentado; no hay data para más)


def _matrix(pmf_h: np.ndarray, pmf_a: np.ndarray) -> np.ndarray:
    m = np.outer(pmf_h, pmf_a)
    s = float(m.sum())
    return m / s if s > 0 else m


def moneyline(pmf_rh: np.ndarray, pmf_ra: np.ndarray) -> dict[str, float]:
    """P(gana local / gana visita) — sin empate (extras 50/50)."""
    m = _matrix(pmf_rh, pmf_ra)
    i = np.arange(m.shape[0]).reshape(-1, 1)
    j = np.arange(m.shape[1]).reshape(1, -1)
    ph = float(m[i > j].sum())
    pt = float(m[i == j].sum())
    home = ph + EXTRA_INNINGS_HOME_P * pt
    return {"home": home, "away": 1.0 - home}


def run_line(pmf_rh: np.ndarray, pmf_ra: np.ndarray, line: float = 1.5) -> dict[str, float]:
    """Run line: local −line / visita +line (margen del local vs el spread)."""
    m = _matrix(pmf_rh, pmf_ra)
    i = np.arange(m.shape[0]).reshape(-1, 1)
    j = np.arange(m.shape[1]).reshape(1, -1)
    margin = i - j
    p_home_cover = float(m[margin > line].sum())
    return {f"home_-{line}": p_home_cover, f"away_+{line}": 1.0 - p_home_cover}


def totals(pmf_h: np.ndarray, pmf_a: np.ndarray, lines: tuple[float, ...]) -> dict:
    total = convolve_pmf(pmf_h, pmf_a)
    return {str(ln): {"over": p_over(total, ln), "under": 1.0 - p_over(total, ln)}
            for ln in lines}


def team_totals(pmf: np.ndarray, lines: tuple[float, ...]) -> dict:
    return {str(ln): {"over": p_over(pmf, ln), "under": 1.0 - p_over(pmf, ln)}
            for ln in lines}


def f5_markets(pmf_f5h: np.ndarray, pmf_f5a: np.ndarray) -> dict:
    """Mercados de las primeras 5 entradas (exactos: familia propia runs_f5).

    El ML F5 en bet365 suele ofrecer empate (3-way) o push en el 2-way — acá damos
    el 3-way (local/empate/visita a 5 entradas), que es el honesto.
    """
    m = _matrix(pmf_f5h, pmf_f5a)
    i = np.arange(m.shape[0]).reshape(-1, 1)
    j = np.arange(m.shape[1]).reshape(1, -1)
    return {
        "ml_3way": {
            "home": float(m[i > j].sum()),
            "tie": float(m[i == j].sum()),
            "away": float(m[i < j].sum()),
        },
        "totales": totals(pmf_f5h, pmf_f5a, (3.5, 4.5, 5.5)),
    }


def project_all_mlb(pmfs: dict[str, np.ndarray]) -> dict:
    """Todos los mercados MLB Tier A desde las pmfs unificadas."""
    rh, ra = pmfs["runs_h"], pmfs["runs_a"]
    hh, ha = pmfs["hits_h"], pmfs["hits_a"]
    fh, fa = pmfs["runs_f5_h"], pmfs["runs_f5_a"]
    return {
        "moneyline": moneyline(rh, ra),
        "run_line": run_line(rh, ra, 1.5),
        "totales": totals(rh, ra, (7.5, 8.5, 9.5, 10.5)),
        "team_total_home": team_totals(rh, (3.5, 4.5, 5.5)),
        "team_total_away": team_totals(ra, (3.5, 4.5, 5.5)),
        "hits_totales": totals(hh, ha, (15.5, 16.5, 17.5)),
        "f5": f5_markets(fh, fa),
    }
