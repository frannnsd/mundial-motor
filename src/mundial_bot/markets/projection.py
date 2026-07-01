"""Proyección de mercados bet365 desde las distribuciones del cerebro unificado.

Capa DETERMINÍSTICA y común a todos los cerebros: no compite, solo traduce.
Entrada: pmfs marginales por cantidad (goals_h, goals_a, corners_h, ...) del
cerebro unificado. Salida: probabilidades de los mercados Tier A.

Horizonte: toda la data de clubes es 90' reglamentario. El parámetro
``horizon`` queda en la interfaz para que la fase Mundial pueda pedir "120"
(prórroga) — la implementación 120' es un TODO explícito (NotImplementedError).

Caveats honestos (documentados también en el reporte):
- El score matrix de goles asume INDEPENDENCIA local/visita (sin la corrección
  tau de Dixon-Coles para 0-0/1-0/0-1/1-1). Sesgo conocido y chico; TODO.
- MT/RF y "mitad con más goles" requieren el modelo de goles con split de
  mitades (la data HTHG/HTAG ya está en el loader) — TODO Fase A2.
"""

from __future__ import annotations

import numpy as np

from mundial_bot.research.distributions import convolve_pmf, p_over

Horizon = str  # "90" | "120"


def _check_horizon(horizon: Horizon) -> None:
    if horizon == "120":
        raise NotImplementedError(
            "Horizonte 120' (prórroga) es TODO: requiere reescalar las tasas a 120 "
            "minutos + el submodelo empate→prórroga→penales (fase Mundial)."
        )
    if horizon != "90":
        raise ValueError(f"Horizonte desconocido: {horizon!r} (usar '90' o '120')")


def score_matrix(pmf_gh: np.ndarray, pmf_ga: np.ndarray, *, horizon: Horizon = "90") -> np.ndarray:
    """Matriz P[i,j] = P(local i, visita j) — independencia (tau DC: TODO)."""
    _check_horizon(horizon)
    m = np.outer(pmf_gh, pmf_ga)
    s = float(m.sum())
    return m / s if s > 0 else m


def one_x_two(
    pmf_gh: np.ndarray, pmf_ga: np.ndarray, *, horizon: Horizon = "90"
) -> dict[str, float]:
    m = score_matrix(pmf_gh, pmf_ga, horizon=horizon)
    i = np.arange(m.shape[0]).reshape(-1, 1)
    j = np.arange(m.shape[1]).reshape(1, -1)
    return {
        "home": float(m[i > j].sum()),
        "draw": float(m[i == j].sum()),
        "away": float(m[i < j].sum()),
    }


def double_chance(pmf_gh, pmf_ga, *, horizon: Horizon = "90") -> dict[str, float]:
    p = one_x_two(pmf_gh, pmf_ga, horizon=horizon)
    return {"1X": p["home"] + p["draw"], "12": p["home"] + p["away"], "X2": p["draw"] + p["away"]}


def total_over_under(pmf_h, pmf_a, line: float, *, horizon: Horizon = "90") -> dict[str, float]:
    """Over/under del TOTAL (goles, córners, tarjetas, remates — cualquier familia)."""
    _check_horizon(horizon)
    total = convolve_pmf(pmf_h, pmf_a)
    over = p_over(total, line)
    return {"over": over, "under": 1.0 - over}

def goal_ranges(pmf_gh, pmf_ga, *, horizon: Horizon = "90") -> dict[str, float]:
    """Rango de goles totales (bandas estándar de bet365: 0-1, 2-3, 4-6, 7+)."""
    _check_horizon(horizon)
    total = convolve_pmf(pmf_gh, pmf_ga)
    def band(lo: int, hi: int | None) -> float:
        return float(total[lo:].sum() if hi is None else total[lo : hi + 1].sum())
    return {"0-1": band(0, 1), "2-3": band(2, 3), "4-6": band(4, 6), "7+": band(7, None)}


def correct_score(
    pmf_gh, pmf_ga, *, max_goals: int = 5, horizon: Horizon = "90"
) -> dict[str, float]:
    """Marcadores exactos hasta max_goals (el resto queda como 'otro')."""
    m = score_matrix(pmf_gh, pmf_ga, horizon=horizon)
    out: dict[str, float] = {}
    for i in range(min(max_goals, m.shape[0] - 1) + 1):
        for j in range(min(max_goals, m.shape[1] - 1) + 1):
            out[f"{i}-{j}"] = float(m[i, j])
    out["otro"] = max(0.0, 1.0 - sum(out.values()))
    return out


def winning_margin(pmf_gh, pmf_ga, *, horizon: Horizon = "90") -> dict[str, float]:
    """Distribución del margen (local − visita): {'-2': p, ..., '0': p, '+1': p, ...}."""
    m = score_matrix(pmf_gh, pmf_ga, horizon=horizon)
    out: dict[str, float] = {}
    n = m.shape[0]
    for d in range(-(n - 1), n):
        p = float(np.trace(m, offset=-d))  # offset -d: filas i = j + d
        if p > 1e-9:
            out[f"{d:+d}" if d else "0"] = p
    return out


def btts(pmf_gh, pmf_ga, *, horizon: Horizon = "90") -> dict[str, float]:
    """Ambos equipos anotan (independencia de marginales)."""
    _check_horizon(horizon)
    yes = (1.0 - float(pmf_gh[0])) * (1.0 - float(pmf_ga[0]))
    return {"yes": yes, "no": 1.0 - yes}


def team_most(pmf_h, pmf_a, *, horizon: Horizon = "90") -> dict[str, float]:
    """Qué equipo tiene MÁS de una cantidad (ej. córners): local/empate/visita."""
    m = score_matrix(pmf_h, pmf_a, horizon=horizon)  # misma matemática que goles
    i = np.arange(m.shape[0]).reshape(-1, 1)
    j = np.arange(m.shape[1]).reshape(1, -1)
    return {
        "home": float(m[i > j].sum()),
        "tie": float(m[i == j].sum()),
        "away": float(m[i < j].sum()),
    }


def both_teams_carded(pmf_yh, pmf_ya, *, horizon: Horizon = "90") -> dict[str, float]:
    """Ambos equipos reciben al menos una tarjeta."""
    _check_horizon(horizon)
    yes = (1.0 - float(pmf_yh[0])) * (1.0 - float(pmf_ya[0]))
    return {"yes": yes, "no": 1.0 - yes}


def any_booking(pmf_yh, pmf_ya, *, horizon: Horizon = "90") -> float:
    """P(alguna amonestación en el partido)."""
    _check_horizon(horizon)
    total = convolve_pmf(pmf_yh, pmf_ya)
    return 1.0 - float(total[0])


def red_card_in_match(pmf_rh, pmf_ra, *, horizon: Horizon = "90") -> float:
    """P(al menos una roja) — base rate baja: intervalo ancho, manejar expectativas."""
    _check_horizon(horizon)
    total = convolve_pmf(pmf_rh, pmf_ra)
    return 1.0 - float(total[0])


def ht_ft(*_args, **_kwargs):
    """Medio tiempo / final — TODO: necesita el modelo de goles con split de mitades
    (HTHG/HTAG ya están en el loader; es la extensión A* del mapa de mercados)."""
    raise NotImplementedError("MT/RF requiere el split de mitades (TODO Fase A2).")


def half_most_goals(*_args, **_kwargs):
    """Mitad con más goles — TODO igual que ht_ft (split de mitades)."""
    raise NotImplementedError("Mitad con más goles requiere el split de mitades (TODO).")


def project_all(unified_pmfs: dict[str, np.ndarray], *, horizon: Horizon = "90") -> dict:
    """Todos los mercados Tier A desde las pmfs del cerebro unificado.

    ``unified_pmfs``: {'goals_h': pmf, 'goals_a': pmf, 'corners_h': ..., ...}
    """
    gh, ga = unified_pmfs["goals_h"], unified_pmfs["goals_a"]
    ch, ca = unified_pmfs["corners_h"], unified_pmfs["corners_a"]
    yh, ya = unified_pmfs["yellows_h"], unified_pmfs["yellows_a"]
    sh, sa = unified_pmfs["shots_h"], unified_pmfs["shots_a"]
    th, ta = unified_pmfs["sot_h"], unified_pmfs["sot_a"]
    rh, ra = unified_pmfs["reds_h"], unified_pmfs["reds_a"]
    return {
        "1x2": one_x_two(gh, ga, horizon=horizon),
        "doble_oportunidad": double_chance(gh, ga, horizon=horizon),
        "goles_ou": {ln: total_over_under(gh, ga, ln, horizon=horizon)
                     for ln in (1.5, 2.5, 3.5)},
        "rango_goles": goal_ranges(gh, ga, horizon=horizon),
        "marcador_exacto": correct_score(gh, ga, horizon=horizon),
        "margen": winning_margin(gh, ga, horizon=horizon),
        "btts": btts(gh, ga, horizon=horizon),
        "corners_ou": {ln: total_over_under(ch, ca, ln, horizon=horizon)
                       for ln in (8.5, 9.5, 10.5, 11.5)},
        "corners_equipo_mas": team_most(ch, ca, horizon=horizon),
        "tarjetas_ou": {ln: total_over_under(yh, ya, ln, horizon=horizon)
                        for ln in (2.5, 3.5, 4.5, 5.5)},
        "ambos_con_tarjeta": both_teams_carded(yh, ya, horizon=horizon),
        "alguna_amonestacion": any_booking(yh, ya, horizon=horizon),
        "roja_en_partido": red_card_in_match(rh, ra, horizon=horizon),
        "remates_ou": {ln: total_over_under(sh, sa, ln, horizon=horizon)
                       for ln in (22.5, 24.5, 26.5)},
        "remates_arco_ou": {ln: total_over_under(th, ta, ln, horizon=horizon)
                            for ln in (7.5, 8.5, 9.5)},
    }
