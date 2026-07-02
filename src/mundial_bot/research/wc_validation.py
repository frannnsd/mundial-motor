"""Validación de los cerebros de SELECCIONES contra los partidos jugados del Mundial.

Setup point-in-time legítimo (Parte 3 del plan):
- El estado camina por TODO el histórico internacional (2022→) en orden cronológico.
- Se puntúan SOLO los partidos del Mundial 2026 ya jugados (``score_filter``), cada
  uno predicho con as_of=kickoff. Los partidos previos del propio torneo SÍ alimentan
  las features de los siguientes (legal y realista).
- El guard anti-leakage corre DENTRO del loop (heredado de run_competition).

Tuning (Parte 2): decay y peso de amistosos se eligen SOLO con esta validación,
sobre una grilla chica (no hay data para más). Adaptaciones selecciones vs clubes:
shrinkage más fuerte (k=8 vs 3: 10-15 partidos/año por equipo), decay largo
(~2 años: los planteles rotan lento entre ciclos), amistosos con peso reducido,
y el bobo/Fano usan SOLO partidos competitivos.

CAVEAT OBLIGATORIO (va en el reporte): la validación son ~72-90 partidos. Es una
muestra CHICA; los pesos son ruidosos. Por eso `unify_nt` usa pesos UNIFORMES entre
los cerebros que le ganan al bobo cuando la diferencia entre ellos no es clara
(gap relativo < UNIFORM_THRESHOLD). El hold-out real del sistema de selecciones es
el FORWARD-TEST de la fase eliminatoria — no existe otro hold-out.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd

from mundial_bot.research.brains import BrainConfig
from mundial_bot.research.competition import (
    QUANTITIES,
    CompetitionResult,
    run_competition,
)

WC_START = pd.Timestamp("2026-06-11")
UNIFORM_THRESHOLD = 0.015  # gap relativo de CRPS bajo el cual los pesos son uniformes

# Config base de selecciones (lo NO tuneado, fijo y documentado).
NT_BASE_CONFIG = BrainConfig(
    halflife_days=730.0,          # se tunea en la grilla
    form_halflife_days=120.0,     # "forma" de selección = últimas ~2 ventanas FIFA
    shrink_k=8.0,                 # shrinkage fuerte: pocas observaciones por equipo
    refit_days=45,
    min_fit_rows=300,
    dumb_competitive_only=True,   # bobo = promedio de COMPETITIVOS point-in-time
    match_type_weights={
        "amistoso": 0.5,          # se tunea en la grilla
        "eliminatoria": 1.0,
        "nations_league": 1.0,
        "continental": 1.2,
        "mundial": 1.2,
        "otro": 0.8,
    },
)

# Grilla chica (4 configs): decay × peso de amistosos. No hay data para más.
TUNING_GRID: list[dict] = [
    {"halflife_days": h, "friendly_weight": f}
    for h in (365.0, 730.0)
    for f in (0.5, 0.75)
]


def _config_from_grid(point: dict) -> BrainConfig:
    weights = dict(NT_BASE_CONFIG.match_type_weights)
    weights["amistoso"] = point["friendly_weight"]
    return replace(
        NT_BASE_CONFIG,
        halflife_days=point["halflife_days"],
        match_type_weights=weights,
    )


def _is_wc_match(row: pd.Series) -> bool:
    return row.get("match_type") == "mundial" and str(row.get("season")) == "2026"


REAL_QUANTITIES = tuple(q for q in QUANTITIES if not q.startswith("reds"))


def _tuning_score(res: CompetitionResult) -> float:
    """Score de un config: media de CRPS_cerebro/CRPS_bobo sobre cantidades reales.

    Promedia los TRES cerebros (no solo el mejor) para no elegir un config que
    beneficia a uno de casualidad — con 70-90 partidos, robustez > pico.
    """
    ratios: list[float] = []
    for q in REAL_QUANTITIES:
        table = res.validation.get(q)
        if not table or "bobo" not in table:
            continue
        bobo = table["bobo"]["crps"]
        for brain in ("A", "B", "C"):
            if table.get(brain, {}).get("crps"):
                ratios.append(table[brain]["crps"] / bobo)
    return float(sum(ratios) / len(ratios)) if ratios else float("inf")


def tune_nt_config(df: pd.DataFrame) -> tuple[BrainConfig, list[dict]]:
    """Elige el config de la grilla con mejor score EN VALIDACIÓN (los partidos WC).

    Devuelve (config elegido, log de la grilla completa para el reporte).
    """
    log: list[dict] = []
    best_cfg, best_score = None, float("inf")
    for point in TUNING_GRID:
        cfg = _config_from_grid(point)
        res = run_competition(df, config=cfg, score_filter=_is_wc_match)
        score = _tuning_score(res)
        log.append({**point, "score": round(score, 5), "n": res.n_scored_validation})
        if score < best_score:
            best_cfg, best_score = cfg, score
    return best_cfg, log


def unify_nt(validation: dict, *, uniform_threshold: float = UNIFORM_THRESHOLD) -> dict:
    """Pesos por cantidad para selecciones, con la regla de MUESTRA CHICA.

    - Elegibles: cerebros con CRPS < CRPS_bobo (perder contra el bobo ⇒ peso 0).
    - Si el gap relativo entre el mejor y el peor elegible es < ``uniform_threshold``
      → pesos UNIFORMES entre elegibles (con ~80 partidos no se distingue más).
    - Si el gap es claro → softmax inverso del CRPS (misma fórmula que clubes).
    - Sin elegibles → el unificado es el bobo.
    """
    weights: dict[str, dict[str, float]] = {}
    for quantity, table in validation.items():
        crps_bobo = table["bobo"]["crps"]
        eligible = {
            b: table[b]["crps"] for b in ("A", "B", "C")
            if table.get(b, {}).get("crps") is not None and table[b]["crps"] < crps_bobo
        }
        if not eligible:
            weights[quantity] = {"bobo": 1.0}
            continue
        best = min(eligible.values())
        worst = max(eligible.values())
        if (worst - best) / best < uniform_threshold:
            w = 1.0 / len(eligible)
            weights[quantity] = {b: w for b in eligible}
            continue
        tau = max(crps_bobo - best, 1e-9) / 3.0
        raw = {b: math.exp(-(c - best) / tau) for b, c in eligible.items()}
        z = sum(raw.values())
        weights[quantity] = {b: v / z for b, v in raw.items()}
    return weights


def run_wc_validation(df: pd.DataFrame | None = None) -> dict:
    """Pipeline completo P2+P3: tuning → competencia final → pesos NT.

    ``df``: tabla de selecciones (collectors.nt_data). Devuelve dict listo para
    el reporte (grilla, config, tabla de validación, pesos, n).
    """
    if df is None:
        from mundial_bot.collectors.nt_data import build_nt_match_table
        df = build_nt_match_table()
    if df.empty:
        return {"error": "tabla de selecciones vacía"}

    cfg, grid_log = tune_nt_config(df)
    final = run_competition(df, config=cfg, score_filter=_is_wc_match)
    weights = unify_nt(final.validation)
    return {
        "n_validation": final.n_scored_validation,
        "grid_log": grid_log,
        "chosen_config": {
            "halflife_days": cfg.halflife_days,
            "form_halflife_days": cfg.form_halflife_days,
            "shrink_k": cfg.shrink_k,
            "match_type_weights": cfg.match_type_weights,
            "dumb_competitive_only": cfg.dumb_competitive_only,
        },
        "validation": {q: final.validation[q] for q in QUANTITIES if q in final.validation},
        "weights": weights,
        "caveat": (
            f"Validación = {final.n_scored_validation} partidos del Mundial (muestra "
            "CHICA; pesos ruidosos — regla uniforme si gap<1.5%). El hold-out real es "
            "el forward-test de la fase eliminatoria."
        ),
    }
