"""La competencia de cerebros: runner walk-forward + métricas + unificación (Fase A).

Reglas innegociables implementadas acá:
- POINT-IN-TIME: same-day batching (se predice todo el día ANTES de revelar nada) y
  `assert_point_in_time()` corre DENTRO del loop, por partido. Si falla, se corta.
- MISMA VARA: los 4 cerebros se puntúan sobre idénticos partidos, cantidades, splits
  y con la misma regla de distribución (`distributions.count_pmf`).
- HOLD-OUT SAGRADO: la temporada `HOLDOUT_SEASON` se puntúa pero sus métricas viven
  en un split aparte que la selección de pesos NUNCA lee. `unify()` recibe SOLO la
  tabla de validación; `evaluate_unified_holdout()` es el único toque al hold-out.
- REPRODUCIBLE: sin RNG (todo determinístico); la config se devuelve en el resultado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mundial_bot.backtest.leakage_guard import assert_point_in_time
from mundial_bot.research.brains import (
    BRAINS,
    FAMILIES,
    BrainConfig,
    LeagueState,
    col_key,
    col_of,
)
from mundial_bot.research.distributions import (
    calibration_table,
    convolve_pmf,
    count_pmf,
    crps_count,
    p_over,
    quantity_grid,
)

HOLDOUT_SEASON = "2324"
WARMUP_SEASON = "1415"  # primera temporada: solo calienta el estado, no se puntúa

# Umbrales estándar de calibración por familia (sobre el TOTAL local+visita).
CALIB_LINES: dict[str, tuple[float, ...]] = {
    "goals": (2.5,),
    "corners": (9.5, 10.5),
    "yellows": (3.5, 4.5),
    "shots": (24.5,),
    "sot": (8.5,),
}

QUANTITIES = tuple(col_key(f, s) for f in FAMILIES for s in ("h", "a"))


@dataclass
class _Acc:
    """Acumulador de métricas de una (cerebro, cantidad) en un split."""

    n: int = 0
    crps_sum: float = 0.0
    abs_sum: float = 0.0
    sq_sum: float = 0.0

    def add(self, crps: float, err: float) -> None:
        self.n += 1
        self.crps_sum += crps
        self.abs_sum += abs(err)
        self.sq_sum += err * err

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "crps": self.crps_sum / self.n,
            "mae": self.abs_sum / self.n,
            "rmse": math.sqrt(self.sq_sum / self.n),
        }


@dataclass
class CompetitionResult:
    config: BrainConfig
    n_scored_validation: int = 0
    n_scored_holdout: int = 0
    # split → cantidad → cerebro → métricas
    validation: dict = field(default_factory=dict)
    holdout: dict = field(default_factory=dict)
    # split → (cerebro, familia, línea) → (tabla fiabilidad, ece)
    calib_validation: dict = field(default_factory=dict)
    # hold-out: por partido, pmfs por cerebro/cantidad (para evaluar el unificado)
    holdout_pmfs: list = field(default_factory=list)


def run_competition(
    df: pd.DataFrame | None = None,
    *,
    config: BrainConfig | None = None,
    player_adjuster=None,
    score_filter=None,
) -> CompetitionResult:
    """Corre los 4 cerebros sobre todos los partidos, point-in-time, guard en el loop.

    ``player_adjuster`` es el ENGANCHE de la Fase B (ajuste por XI/jugadores): si se
    pasa, se aplica a las predicciones de cada cerebro antes de puntuar. Hoy nadie
    lo implementa (default None = no-op).

    ``score_filter``: callable(row) -> bool. Si se pasa, SOLO se puntúan las filas
    donde devuelve True (el resto igual alimenta el estado — walk-forward completo).
    Lo usa la validación de selecciones: el estado camina por TODO el histórico
    internacional pero solo se puntúan los partidos del Mundial. Default None =
    comportamiento de clubes sin cambios.
    """
    if df is None:
        from mundial_bot.collectors.football_data import load_football_stats
        df = load_football_stats()
    cfg = config or BrainConfig()
    res = CompetitionResult(config=cfg)

    acc: dict[tuple[str, str, str], _Acc] = {}   # (split, quantity, brain)
    calib: dict[tuple[str, str, float], list] = {}  # (brain, family, line) → pairs (validación)

    for _, ldf in df.groupby("league"):
        ldf = ldf.sort_values("date").reset_index(drop=True)
        state = LeagueState(cfg)
        n = len(ldf)
        i = 0
        while i < n:
            day = ldf["date"].iloc[i]
            j = i
            while j < n and ldf["date"].iloc[j] == day:
                j += 1
            prior = ldf.iloc[:i]  # SOLO fecha < day (df ordenado): sin same-day leakage

            for k in range(i, j):
                # INNEGOCIABLE: guard DENTRO del loop, por partido. Si falla, corta.
                assert_point_in_time(prior, day, label=f"brains:{ldf['league'].iloc[k]}")
                row = ldf.iloc[k]
                preds = state.predict(row, day)
                if player_adjuster is not None:
                    preds = {
                        b: player_adjuster(row["home_team"], row["away_team"], p)
                        for b, p in preds.items()
                    }
                season = row["season"]
                if season == WARMUP_SEASON:
                    continue  # calienta el estado (reveal abajo) pero no puntúa
                if score_filter is not None and not score_filter(row):
                    continue  # fuera del set a puntuar (igual alimenta el estado)
                split = "holdout" if season == HOLDOUT_SEASON else "validation"
                _score_match(row, preds, split, acc, calib, res)

            for k in range(i, j):
                state.reveal(ldf.iloc[k], day)
            state.end_day(day)
            i = j

    for (split, quantity, brain), a in acc.items():
        target = res.validation if split == "validation" else res.holdout
        target.setdefault(quantity, {})[brain] = a.summary()
    for (brain, family, line), pairs in calib.items():
        res.calib_validation[(brain, family, line)] = calibration_table(pairs)
    res.n_scored_validation = max(
        (m["n"] for q in res.validation.values() for m in q.values()), default=0
    )
    res.n_scored_holdout = max(
        (m["n"] for q in res.holdout.values() for m in q.values()), default=0
    )
    return res


def _score_match(row, preds, split, acc, calib, res) -> None:
    """Puntúa un partido para los 4 cerebros (CRPS/MAE) + calibración + pmfs hold-out."""
    pmf_store: dict[str, dict[str, np.ndarray]] = {} if split == "holdout" else None
    actuals: dict[str, int] = {}

    for family in FAMILIES:
        k_max = quantity_grid(family)
        for side in ("h", "a"):
            q = col_key(family, side)
            actual = int(row[col_of(family, side)])
            actuals[q] = actual
            for brain in BRAINS:
                mean, var = preds[brain][q]
                pmf = count_pmf(mean, var, k_max)
                a = acc.setdefault((split, q, brain), _Acc())
                a.add(crps_count(pmf, actual), mean - actual)
                if pmf_store is not None:
                    pmf_store.setdefault(q, {})[brain] = pmf

        # Calibración sobre el TOTAL (validación solamente: acá se compara/elige).
        if split == "validation" and family in CALIB_LINES:
            total_actual = int(row[col_of(family, "h")]) + int(row[col_of(family, "a")])
            for brain in BRAINS:
                mh, vh = preds[brain][col_key(family, "h")]
                ma, va = preds[brain][col_key(family, "a")]
                total = convolve_pmf(count_pmf(mh, vh, k_max), count_pmf(ma, va, k_max))
                for line in CALIB_LINES[family]:
                    calib.setdefault((brain, family, line), []).append(
                        (p_over(total, line), total_actual > line)
                    )

    if pmf_store is not None:
        res.holdout_pmfs.append({"pmfs": pmf_store, "actuals": actuals,
                                 "match_id": row["match_id"]})


# ---------------------------------------------------------------------------
# UNIFICACIÓN — pesos derivados SOLO de la tabla de validación
# ---------------------------------------------------------------------------

def unify(validation: dict) -> dict[str, dict[str, float]]:
    """Pesos por cantidad desde la evidencia de VALIDACIÓN (nunca del hold-out).

    Fórmula (documentada en el reporte):
      - Elegibles: cerebros con CRPS_validación < CRPS_bobo en esa cantidad.
        Un cerebro que pierde contra el bobo pesa 0 en esa cantidad.
      - w_i ∝ exp(−Δ_i / τ), Δ_i = CRPS_i − CRPS_mejor,
        τ = max(CRPS_bobo − CRPS_mejor, 1e-9) / 3
        (a τ del bobo el peso cae a e⁻³ ≈ 0.05 del mejor).
      - Si NINGÚN cerebro le gana al bobo → el unificado ES el bobo (peso 1).
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
        tau = max(crps_bobo - best, 1e-9) / 3.0
        raw = {b: math.exp(-(c - best) / tau) for b, c in eligible.items()}
        z = sum(raw.values())
        weights[quantity] = {b: w / z for b, w in raw.items()}
    return weights


def evaluate_unified_holdout(res: CompetitionResult, weights: dict) -> dict:
    """ÚNICO toque al hold-out: evalúa el cerebro unificado (mixtura de pmfs).

    La pmf unificada de cada cantidad es Σ w_b · pmf_b (mixtura — conserva la
    calibración de los componentes). Devuelve métricas por cantidad + pares de
    calibración de mercados proyectados (sanity check de la proyección).
    """
    acc: dict[str, _Acc] = {}
    market_pairs: dict[str, list] = {
        "over_2.5_goles": [], "over_9.5_corners": [], "over_3.5_amarillas": [], "btts": [],
    }
    for rec in res.holdout_pmfs:
        pmfs, actuals = rec["pmfs"], rec["actuals"]
        unified: dict[str, np.ndarray] = {}
        for q, by_brain in pmfs.items():
            w = weights[q]
            mix = None
            for b, wb in w.items():
                mix = by_brain[b] * wb if mix is None else mix + by_brain[b] * wb
            unified[q] = mix
            a = acc.setdefault(q, _Acc())
            mean = float(np.dot(np.arange(len(mix)), mix))
            a.add(crps_count(mix, actuals[q]), mean - actuals[q])

        # Sanity de proyección: mercados desde las pmfs unificadas vs lo que pasó.
        tg = convolve_pmf(unified["goals_h"], unified["goals_a"])
        tc = convolve_pmf(unified["corners_h"], unified["corners_a"])
        ty = convolve_pmf(unified["yellows_h"], unified["yellows_a"])
        ag, ac_, ay = (actuals["goals_h"] + actuals["goals_a"],
                       actuals["corners_h"] + actuals["corners_a"],
                       actuals["yellows_h"] + actuals["yellows_a"])
        market_pairs["over_2.5_goles"].append((p_over(tg, 2.5), ag > 2.5))
        market_pairs["over_9.5_corners"].append((p_over(tc, 9.5), ac_ > 9.5))
        market_pairs["over_3.5_amarillas"].append((p_over(ty, 3.5), ay > 3.5))
        p_btts = (1.0 - float(unified["goals_h"][0])) * (1.0 - float(unified["goals_a"][0]))
        market_pairs["btts"].append(
            (p_btts, actuals["goals_h"] >= 1 and actuals["goals_a"] >= 1)
        )

    return {
        "metrics": {q: a.summary() for q, a in acc.items()},
        "market_calibration": {
            m: calibration_table(pairs) for m, pairs in market_pairs.items()
        },
    }
