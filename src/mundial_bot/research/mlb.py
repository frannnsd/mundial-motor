"""Competencia de cerebros MLB (M1): carreras, hits y carreras-F5, point-in-time.

Misma disciplina que la competencia de fútbol (research/competition.py):
- MISMA VARA: los 4 cerebros comparten `distributions.count_pmf` y el scoring CRPS.
- POINT-IN-TIME: same-day batching + `assert_point_in_time` DENTRO del loop.
- HOLD-OUT SAGRADO: temporada 2025 completa; el tuning (grilla chica) y los pesos
  salen SOLO de validación 2016-2024. 2026 (en curso) no se puntúa: es el
  forward-test vivo del M4. Warm-up: 2015.
- Sin RNG; config logueada.

Lo distinto de béisbol (y por qué):
- El PITCHER ABRIDOR es el factor #1: el cerebro B mezcla la defensa del equipo con
  la calidad del abridor rival (proxy point-in-time: lo que le anotaron al equipo en
  los starts previos de ese pitcher — exacto para F5, razonable para el total).
- PARQUE: factor decaído por venue (Coors ≠ Petco), aplicado √ a cada lado.
- Sin empates ni localía simétrica: el local batea segundo (ventaja chica real).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mundial_bot.backtest.leakage_guard import assert_point_in_time
from mundial_bot.research.brains import _Decayed, _fit_poisson_irls, _Moments
from mundial_bot.research.competition import _Acc, unify
from mundial_bot.research.distributions import (
    calibration_table,
    convolve_pmf,
    count_pmf,
    crps_count,
    p_over,
)

FAMILIES_MLB = ("runs", "hits", "runs_f5")
BRAINS = ("A", "B", "C", "bobo")
GRID_MLB = {"runs": 22, "hits": 32, "runs_f5": 16}

WARMUP_SEASON = "2015"
HOLDOUT_SEASON = "2025"
LIVE_SEASON = "2026"  # no se puntúa: forward-test del pipeline vivo


@dataclass
class MlbConfig:
    """Config de la competencia MLB (queda logueada en el reporte)."""

    halflife_days: float = 270.0      # decay de tasas de equipo/pitcher/parque
    form_halflife_days: float = 40.0  # forma corta (feature del GLM)
    shrink_k: float = 10.0            # partidos-equivalentes hacia la media de liga
    starter_shrink_k: float = 4.0     # starts-equivalentes (los pitchers tienen pocos)
    starter_weight: float = 0.6       # peso del abridor vs defensa del equipo (cerebro B)
    park_clip: tuple[float, float] = (0.80, 1.25)
    refit_days: int = 30
    min_fit_rows: int = 800
    l2: float = 1e-3
    var_floor_fano: float = 1.0


def col_of(family: str, side: str) -> str:
    return f"{family}_{side}"


QUANTITIES_MLB = tuple(col_of(f, s) for f in FAMILIES_MLB for s in ("h", "a"))
N_FEATURES = 7  # [1, home, off_rel, def_rel, starter_rel, park, rest]


@dataclass
class _TeamState:
    off: dict[str, _Decayed] = field(default_factory=dict)   # lo que anota
    deff: dict[str, _Decayed] = field(default_factory=dict)  # lo que le anotan
    form: dict[str, _Decayed] = field(default_factory=dict)
    last_game: pd.Timestamp | None = None

    def dec(self, store: dict[str, _Decayed], family: str) -> _Decayed:
        if family not in store:
            store[family] = _Decayed()
        return store[family]


class MlbState:
    """Estado incremental point-in-time: equipos, pitchers, parques, liga."""

    def __init__(self, cfg: MlbConfig | None = None):
        self.cfg = cfg or MlbConfig()
        self.teams: dict[str, _TeamState] = {}
        # pitcher_id → familia → _Decayed de lo PERMITIDO por partido en sus starts.
        self.starters: dict[int, dict[str, _Decayed]] = {}
        # venue → _Decayed del TOTAL del partido (para el factor de parque).
        self.parks: dict[str, dict[str, _Decayed]] = {}
        self.lg: dict[tuple[str, str], _Moments] = {}                 # (familia, lado)
        self.lg_season: dict[tuple[str, str, str], _Moments] = {}     # + temporada (bobo)
        self._c_x: dict[str, list[np.ndarray]] = {f: [] for f in FAMILIES_MLB}
        self._c_y: dict[str, list[float]] = {f: [] for f in FAMILIES_MLB}
        self._c_beta: dict[str, np.ndarray | None] = {f: None for f in FAMILIES_MLB}
        self._c_last_fit: pd.Timestamp | None = None
        self._pending: list[tuple[str, np.ndarray, float]] = []

    # ---------- helpers ----------

    def _team(self, name: str) -> _TeamState:
        if name not in self.teams:
            self.teams[name] = _TeamState()
        return self.teams[name]

    def _mom(self, family: str, side: str) -> _Moments:
        key = (family, side)
        if key not in self.lg:
            self.lg[key] = _Moments()
        return self.lg[key]

    def _mom_season(self, season: str, family: str, side: str) -> _Moments:
        key = (season, family, side)
        if key not in self.lg_season:
            self.lg_season[key] = _Moments()
        return self.lg_season[key]

    def _lg_all(self, family: str) -> float:
        h, a = self._mom(family, "h"), self._mom(family, "a")
        n = h.n + a.n
        return (h.total + a.total) / n if n >= 1 else {"runs": 4.5, "hits": 8.4,
                                                       "runs_f5": 2.5}[family]

    def _fano(self, family: str, side: str) -> float:
        mom = self._mom(family, side)
        m, v = mom.mean, mom.var
        if m is None or v is None or m <= 0:
            return self.cfg.var_floor_fano
        return max(v / m, self.cfg.var_floor_fano)

    def _shrunk(self, dec: _Decayed, at: pd.Timestamp, lg: float, k: float) -> float:
        raw = dec.rate(at, self.cfg.halflife_days, lg)
        n = dec.eff_n(at, self.cfg.halflife_days)
        return (n * raw + k * lg) / (n + k)

    def _park_factor(self, venue: str, at: pd.Timestamp) -> float:
        """Total decaído en el venue / total de liga, clippeado (1.0 sin datos)."""
        dec = self.parks.get(venue, {}).get("runs")
        lg_total = self._lg_all("runs") * 2
        if dec is None or lg_total <= 0:
            return 1.0
        n = dec.eff_n(at, self.cfg.halflife_days)
        if n < 10:
            return 1.0
        raw = dec.rate(at, self.cfg.halflife_days, lg_total) / lg_total
        lo, hi = self.cfg.park_clip
        return min(max(raw, lo), hi)

    def _starter_allowed(self, pid: int | None, family: str, at: pd.Timestamp,
                         team_def: float, lg: float) -> float:
        """Permitido/partido en los starts del pitcher, shrunk hacia la DEFENSA del
        equipo (si el pitcher tiene pocos starts, manda el equipo)."""
        if pid is None or pid not in self.starters:
            return team_def
        dec = self.starters[pid].get(family)
        if dec is None:
            return team_def
        raw = dec.rate(at, self.cfg.halflife_days, team_def)
        n = dec.eff_n(at, self.cfg.halflife_days)
        k = self.cfg.starter_shrink_k
        _ = lg
        return (n * raw + k * team_def) / (n + k)

    # ---------- predicción (no muta estado del partido) ----------

    def predict(
        self, row: pd.Series, day: pd.Timestamp
    ) -> dict[str, dict[str, tuple[float, float]]]:
        cfg = self.cfg
        home, away, season = row["home_team"], row["away_team"], row["season"]
        th, ta = self._team(home), self._team(away)
        park = self._park_factor(row.get("venue") or "", day)
        park_side = math.sqrt(park)
        out: dict[str, dict[str, tuple[float, float]]] = {b: {} for b in BRAINS}

        for family in FAMILIES_MLB:
            lg = self._lg_all(family)
            for side, _team, _opp, tstate, ostate, starter_opp in (
                ("h", home, away, th, ta, row.get("starter_a_id")),
                ("a", away, home, ta, th, row.get("starter_h_id")),
            ):
                lg_side_mom = self._mom(family, side)
                lg_side = lg_side_mom.mean if lg_side_mom.mean is not None else lg
                fano = self._fano(family, side)
                home_adj = (lg_side / lg) if lg > 0 else 1.0

                off = self._shrunk(tstate.dec(tstate.off, family), day, lg, cfg.shrink_k)
                deff = self._shrunk(ostate.dec(ostate.deff, family), day, lg, cfg.shrink_k)
                sid = None if pd.isna(starter_opp) else int(starter_opp)
                st_allowed = self._starter_allowed(sid, family, day, deff, lg)

                # A — equipo puro: ofensiva + defensa rival + parque + localía.
                mean_a = max(0.5 * (off + deff) * park_side * home_adj, 1e-6)
                out["A"][col_of(family, side)] = (mean_a, mean_a * fano)

                # B — matchup con ABRIDOR: la defensa rival se mezcla con el pitcher.
                w = cfg.starter_weight
                def_eff = w * st_allowed + (1 - w) * deff
                mean_b = max(0.5 * (off + def_eff) * park_side * home_adj, 1e-6)
                out["B"][col_of(family, side)] = (mean_b, mean_b * fano)

                # bobo — media/var de la liga-TEMPORADA hasta hoy.
                ms = self._mom_season(season, family, side)
                if ms.n >= 60 and ms.mean is not None:
                    mean_d, var_d = ms.mean, ms.var or (ms.mean * fano)
                else:
                    mean_d, var_d = lg_side, lg_side * fano
                out["bobo"][col_of(family, side)] = (max(mean_d, 1e-6), max(var_d, 1e-9))

                # C — GLM Poisson con las mismas señales como features.
                rest = 3.0
                if tstate.last_game is not None:
                    rest = float(min(max((day - tstate.last_game).days, 0), 10))
                eps = 0.25
                x = np.array([
                    1.0,
                    1.0 if side == "h" else 0.0,
                    math.log((off + eps) / (lg + eps)),
                    math.log((deff + eps) / (lg + eps)),
                    math.log((st_allowed + eps) / (deff + eps)),
                    math.log(park),
                    math.log((rest + 1) / 2.0),
                ])
                beta = self._c_beta[family]
                if beta is not None:
                    mu = float(np.exp(np.clip(x @ beta, -10.0, 5.0)))
                    mean_c = min(max(mu, 1e-6), lg * 6 + 5)
                else:
                    mean_c = mean_a
                out["C"][col_of(family, side)] = (mean_c, mean_c * fano)

                actual = row.get(col_of(family, side))
                if actual is not None and not pd.isna(actual):
                    self._pending.append((family, x, float(actual)))

        return out

    # ---------- revelado + cierre de día ----------

    def reveal(self, row: pd.Series, day: pd.Timestamp) -> None:
        cfg = self.cfg
        home, away, season = row["home_team"], row["away_team"], row["season"]
        th, ta = self._team(home), self._team(away)
        venue = row.get("venue") or ""
        for family in FAMILIES_MLB:
            vh, va = float(row[col_of(family, "h")]), float(row[col_of(family, "a")])
            th.dec(th.off, family).update(vh, day, cfg.halflife_days)
            th.dec(th.deff, family).update(va, day, cfg.halflife_days)
            ta.dec(ta.off, family).update(va, day, cfg.halflife_days)
            ta.dec(ta.deff, family).update(vh, day, cfg.halflife_days)
            th.dec(th.form, family).update(vh, day, cfg.form_halflife_days)
            ta.dec(ta.form, family).update(va, day, cfg.form_halflife_days)
            self._mom(family, "h").add(vh)
            self._mom(family, "a").add(va)
            self._mom_season(season, family, "h").add(vh)
            self._mom_season(season, family, "a").add(va)
            # abridores: lo que anotó el RIVAL en el start de cada uno.
            for pid_raw, allowed in ((row.get("starter_h_id"), va),
                                     (row.get("starter_a_id"), vh)):
                if pid_raw is not None and not pd.isna(pid_raw):
                    pid = int(pid_raw)
                    self.starters.setdefault(pid, {}).setdefault(
                        family, _Decayed()
                    ).update(allowed, day, cfg.halflife_days)
        self.parks.setdefault(venue, {}).setdefault("runs", _Decayed()).update(
            float(row["runs_h"]) + float(row["runs_a"]), day, cfg.halflife_days
        )
        th.last_game = day
        ta.last_game = day

    def end_day(self, day: pd.Timestamp) -> None:
        for family, x, y in self._pending:
            self._c_x[family].append(x)
            self._c_y[family].append(y)
        self._pending.clear()
        cfg = self.cfg
        if self._c_last_fit is not None and (day - self._c_last_fit).days < cfg.refit_days:
            return
        fitted = False
        for family in FAMILIES_MLB:
            y = self._c_y[family]
            if len(y) < cfg.min_fit_rows:
                continue
            beta = _fit_poisson_irls(np.vstack(self._c_x[family]),
                                     np.asarray(y, dtype=float), l2=cfg.l2)
            if beta is not None:
                self._c_beta[family] = beta
                fitted = True
        if fitted or self._c_last_fit is None:
            self._c_last_fit = day


# ---------------------------------------------------------------------------
# Runner walk-forward (guard EN el loop) + evaluación del unificado
# ---------------------------------------------------------------------------

# Calibración estándar sobre mercados MLB (validación).
CALIB_MLB = {
    "total_o8.5": ("runs", 8.5),
    "f5_o4.5": ("runs_f5", 4.5),
}


def run_mlb_competition(df: pd.DataFrame, *, config: MlbConfig | None = None) -> dict:
    cfg = config or MlbConfig()
    df = df.sort_values("date").reset_index(drop=True)
    state = MlbState(cfg)
    acc: dict[tuple[str, str, str], _Acc] = {}
    calib: dict[tuple[str, str], list] = {}     # (cerebro, mercado) → pares (validación)
    ml_pairs: dict[str, list] = {b: [] for b in BRAINS}  # P(gana local) vs real (validación)
    holdout_pmfs: list[dict] = []

    n = len(df)
    i = 0
    while i < n:
        day = df["date"].iloc[i]
        j = i
        while j < n and df["date"].iloc[j] == day:
            j += 1
        prior = df.iloc[:i]
        for k in range(i, j):
            assert_point_in_time(prior, day, label="mlb")  # INNEGOCIABLE, por partido
            row = df.iloc[k]
            preds = state.predict(row, day)
            season = row["season"]
            if season in (WARMUP_SEASON, LIVE_SEASON):
                continue
            split = "holdout" if season == HOLDOUT_SEASON else "validation"
            _score_mlb(row, preds, split, acc, calib, ml_pairs, holdout_pmfs)
        for k in range(i, j):
            state.reveal(df.iloc[k], day)
        state.end_day(day)
        i = j

    validation: dict = {}
    holdout: dict = {}
    for (split, q, brain), a in acc.items():
        (validation if split == "validation" else holdout).setdefault(q, {})[brain] = a.summary()
    return {
        "config": cfg,
        "validation": validation,
        "holdout": holdout,
        "calib_validation": {f"{b}|{m}": calibration_table(pairs)[1]
                             for (b, m), pairs in calib.items()},
        "ml_calibration": {b: calibration_table(pairs)[1] for b, pairs in ml_pairs.items()
                           if pairs},
        "holdout_pmfs": holdout_pmfs,
        "n_validation": max((m["n"] for q in validation.values() for m in q.values()),
                            default=0),
        "n_holdout": max((m["n"] for q in holdout.values() for m in q.values()), default=0),
    }


def _ml_prob_home(pmf_h: np.ndarray, pmf_a: np.ndarray) -> float:
    """P(gana el local) desde las pmfs de carreras; el empate 9 entradas se parte
    50/50 (extra innings ~ moneda; documentado, no tuneado)."""
    m = np.outer(pmf_h, pmf_a)
    m = m / m.sum()
    i = np.arange(m.shape[0]).reshape(-1, 1)
    jj = np.arange(m.shape[1]).reshape(1, -1)
    p_home = float(m[i > jj].sum())
    p_tie = float(m[i == jj].sum())
    return p_home + 0.5 * p_tie


def _score_mlb(row, preds, split, acc, calib, ml_pairs, holdout_pmfs) -> None:
    pmf_store: dict[str, dict[str, np.ndarray]] = {} if split == "holdout" else None
    actuals: dict[str, int] = {}
    for family in FAMILIES_MLB:
        k_max = GRID_MLB[family]
        for side in ("h", "a"):
            q = col_of(family, side)
            actual = int(row[q])
            actuals[q] = actual
            for brain in BRAINS:
                mean, var = preds[brain][q]
                pmf = count_pmf(mean, var, k_max)
                acc.setdefault((split, q, brain), _Acc()).add(
                    crps_count(pmf, actual), mean - actual
                )
                if pmf_store is not None:
                    pmf_store.setdefault(q, {})[brain] = pmf
    if split == "validation":
        for name, (family, line) in CALIB_MLB.items():
            total = int(row[col_of(family, "h")]) + int(row[col_of(family, "a")])
            k_max = GRID_MLB[family]
            for brain in BRAINS:
                mh, vh = preds[brain][col_of(family, "h")]
                ma, va = preds[brain][col_of(family, "a")]
                tp = convolve_pmf(count_pmf(mh, vh, k_max), count_pmf(ma, va, k_max))
                calib.setdefault((brain, name), []).append((p_over(tp, line), total > line))
        home_won = int(row["runs_h"]) > int(row["runs_a"])
        for brain in BRAINS:
            mh, vh = preds[brain]["runs_h"]
            ma, va = preds[brain]["runs_a"]
            p = _ml_prob_home(count_pmf(mh, vh, GRID_MLB["runs"]),
                              count_pmf(ma, va, GRID_MLB["runs"]))
            ml_pairs[brain].append((p, home_won))
    if pmf_store is not None:
        holdout_pmfs.append({"pmfs": pmf_store, "actuals": actuals})


def evaluate_unified_mlb(result: dict, weights: dict) -> dict:
    """ÚNICO toque al hold-out 2025: mixtura de pmfs con los pesos de validación."""
    acc: dict[str, _Acc] = {}
    market_pairs = {"total_o8.5": [], "f5_o4.5": [], "ml_home": []}
    for rec in result["holdout_pmfs"]:
        pmfs, actuals = rec["pmfs"], rec["actuals"]
        uni: dict[str, np.ndarray] = {}
        for q, by_brain in pmfs.items():
            mix = None
            for b, wb in weights[q].items():
                mix = by_brain[b] * wb if mix is None else mix + by_brain[b] * wb
            uni[q] = mix
            mean = float(np.dot(np.arange(len(mix)), mix))
            acc.setdefault(q, _Acc()).add(crps_count(mix, actuals[q]), mean - actuals[q])
        tr = convolve_pmf(uni["runs_h"], uni["runs_a"])
        tf5 = convolve_pmf(uni["runs_f5_h"], uni["runs_f5_a"])
        market_pairs["total_o8.5"].append(
            (p_over(tr, 8.5), actuals["runs_h"] + actuals["runs_a"] > 8.5))
        market_pairs["f5_o4.5"].append(
            (p_over(tf5, 4.5), actuals["runs_f5_h"] + actuals["runs_f5_a"] > 4.5))
        market_pairs["ml_home"].append(
            (_ml_prob_home(uni["runs_h"], uni["runs_a"]),
             actuals["runs_h"] > actuals["runs_a"]))
    return {
        "metrics": {q: a.summary() for q, a in acc.items()},
        "market_calibration": {m: calibration_table(pairs)
                               for m, pairs in market_pairs.items()},
    }


__all__ = [
    "FAMILIES_MLB", "QUANTITIES_MLB", "MlbConfig", "MlbState",
    "run_mlb_competition", "evaluate_unified_mlb", "unify",
]
