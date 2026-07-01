"""Los cuatro competidores de la Fase A — todos point-in-time por construcción.

Cada cerebro produce, por partido y por cantidad, una predicción (media, varianza)
que `distributions.count_pmf` convierte en pmf con la MISMA regla para todos.

Point-in-time: el estado se actualiza SOLO con partidos ya revelados (el runner
predice todo el día antes de actualizar, igual que el CLV backtest). Las features
de entrenamiento del Cerebro C se capturan EN el momento de la predicción — el set
de entrenamiento es un replay exacto de lo que se sabía antes de cada kickoff.

Cerebros:
  A — Tasas históricas: media decaída de lo que el equipo genera + lo que el rival
      concede, ajuste por localía, dispersión NegBin (Fano de liga por lado).
  B — Matchup multiplicativo: ataque × concesión / media de liga (estilo Dixon-Coles
      aplicado a conteos), misma dispersión.
  C — GLM Poisson (IRLS propio, sin dependencias): localía + tasas relativas +
      descanso + forma + H2H. Refit periódico con SOLO datos pasados.
  bobo — Media (y varianza) corriente de la liga-TEMPORADA hasta ese partido.

Sin RNG en ningún cerebro: todo es determinístico (reproducible por construcción).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Familias de cantidades (cada una tiene lado _h y _a en el dataset).
FAMILIES = ("goals", "corners", "yellows", "shots", "sot", "reds")
BRAINS = ("A", "B", "C", "bobo")

# Columna del dataset para (familia, lado).
def col_of(family: str, side: str) -> str:
    if family == "goals":
        return "home_score" if side == "h" else "away_score"
    return f"{family}_{side}"


@dataclass
class BrainConfig:
    """Configuración de la competencia (se loguea en el reporte — reproducible)."""

    halflife_days: float = 300.0     # decay de las tasas largas (A/B/features C)
    form_halflife_days: float = 45.0 # decay de la forma corta (feature C)
    shrink_k: float = 3.0            # shrinkage de tasas hacia la media de liga
    refit_days: int = 30             # cadencia de refit del GLM (Cerebro C)
    min_fit_rows: int = 500          # filas mínimas para el primer fit del GLM
    l2: float = 1e-3                 # regularización ridge del IRLS
    var_floor_fano: float = 1.0      # piso del Fano (var >= media → Poisson)


@dataclass
class _Decayed:
    """Suma/conteo con decaimiento exponencial por días (estado incremental)."""

    s: float = 0.0
    w: float = 0.0
    last: pd.Timestamp | None = None

    def _factor(self, at: pd.Timestamp, halflife: float) -> float:
        if self.last is None:
            return 1.0
        days = max((at - self.last).days, 0)
        return float(0.5 ** (days / halflife))

    def rate(self, at: pd.Timestamp, halflife: float, default: float) -> float:
        if self.last is None or self.w <= 1e-9:
            return default
        f = self._factor(at, halflife)
        w = self.w * f
        return (self.s * f / w) if w > 1e-9 else default

    def eff_n(self, at: pd.Timestamp, halflife: float) -> float:
        return self.w * self._factor(at, halflife) if self.last is not None else 0.0

    def update(self, value: float, at: pd.Timestamp, halflife: float) -> None:
        f = self._factor(at, halflife)
        self.s = self.s * f + value
        self.w = self.w * f + 1.0
        self.last = at


@dataclass
class _Moments:
    """Media/varianza corrientes (para Fano de liga y para el bobo)."""

    n: float = 0.0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1.0
        self.total += x
        self.total_sq += x * x

    @property
    def mean(self) -> float | None:
        return self.total / self.n if self.n >= 1 else None

    @property
    def var(self) -> float | None:
        if self.n < 2:
            return None
        m = self.total / self.n
        return max(self.total_sq / self.n - m * m, 1e-9)


@dataclass
class _TeamState:
    for_: dict[str, _Decayed] = field(default_factory=dict)
    conc: dict[str, _Decayed] = field(default_factory=dict)
    form: dict[str, _Decayed] = field(default_factory=dict)
    last_match: pd.Timestamp | None = None

    def dec(self, store: dict[str, _Decayed], family: str) -> _Decayed:
        if family not in store:
            store[family] = _Decayed()
        return store[family]


def _fit_poisson_irls(
    x: np.ndarray, y: np.ndarray, *, l2: float, iters: int = 30
) -> np.ndarray | None:
    """GLM Poisson log-link por IRLS con ridge. Devuelve beta o None si no converge."""
    n, p = x.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(float(y.mean()), 1e-3))
    eye = np.eye(p)
    for _ in range(iters):
        eta = np.clip(x @ beta, -12.0, 6.0)
        mu = np.exp(eta)
        w = mu
        z = eta + (y - mu) / np.maximum(mu, 1e-9)
        a = x.T @ (w[:, None] * x) + l2 * n * eye
        b = x.T @ (w * z)
        try:
            new = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new)):
            return None
        if float(np.max(np.abs(new - beta))) < 1e-9:
            beta = new
            break
        beta = new
    return beta


N_FEATURES = 7  # [1, home, log_att_rel, log_conc_rel, rest, form_rel, h2h_rel]


class LeagueState:
    """Estado incremental de UNA liga: predice (los 4 cerebros) y luego revela.

    Contrato point-in-time: `predict(row, day)` NO muta el estado con datos del
    partido; `reveal(row, day)` (llamado por el runner recién después de predecir
    todo el día) incorpora el resultado.
    """

    def __init__(self, config: BrainConfig | None = None):
        self.cfg = config or BrainConfig()
        self.teams: dict[str, _TeamState] = {}
        # Momentos de liga por (familia, lado) — all-time, para Fano y fallbacks.
        self.lg: dict[tuple[str, str], _Moments] = {}
        # Momentos de liga-TEMPORADA por (season, familia, lado) — para el bobo.
        self.lg_season: dict[tuple[str, str, str], _Moments] = {}
        # H2H direccional: (equipo, rival) → familia → _Decayed de lo producido.
        self.h2h: dict[tuple[str, str], dict[str, _Decayed]] = {}
        # Cerebro C: filas de entrenamiento y modelos por familia.
        self._c_x: dict[str, list[np.ndarray]] = {f: [] for f in FAMILIES}
        self._c_y: dict[str, list[float]] = {f: [] for f in FAMILIES}
        self._c_beta: dict[str, np.ndarray | None] = {f: None for f in FAMILIES}
        self._c_last_fit: pd.Timestamp | None = None
        self._pending: list[tuple[str, str, np.ndarray, float]] = []  # (family, side, x, y)

    # ---------- helpers de estado ----------

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

    def _lg_side_mean(self, family: str, side: str, default: float) -> float:
        m = self._mom(family, side).mean
        return m if m is not None else default

    def _lg_all_mean(self, family: str) -> float:
        h, a = self._mom(family, "h"), self._mom(family, "a")
        n = h.n + a.n
        return (h.total + a.total) / n if n >= 1 else 1.0

    def _fano(self, family: str, side: str) -> float:
        mom = self._mom(family, side)
        m, v = mom.mean, mom.var
        if m is None or v is None or m <= 0:
            return self.cfg.var_floor_fano
        return max(v / m, self.cfg.var_floor_fano)

    def _shrunk_rate(self, dec: _Decayed, at: pd.Timestamp, lg_mean: float) -> float:
        raw = dec.rate(at, self.cfg.halflife_days, lg_mean)
        n = dec.eff_n(at, self.cfg.halflife_days)
        k = self.cfg.shrink_k
        return (n * raw + k * lg_mean) / (n + k)

    # ---------- predicción (NO muta estado con el partido) ----------

    def predict(
        self, row: pd.Series, day: pd.Timestamp
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Predicciones de los 4 cerebros para las 12 cantidades del partido.

        Devuelve {cerebro: {cantidad(ej 'corners_h'): (media, varianza)}}.
        También captura (en pending) las features del Cerebro C para entrenar
        cuando el partido se revele — replay exacto point-in-time.
        """
        home, away, season = row["home_team"], row["away_team"], row["season"]
        th, ta = self._team(home), self._team(away)
        out: dict[str, dict[str, tuple[float, float]]] = {b: {} for b in BRAINS}

        for family in FAMILIES:
            lg_all = self._lg_all_mean(family)
            for side, team, opp, tstate, ostate in (
                ("h", home, away, th, ta),
                ("a", away, home, ta, th),
            ):
                lg_side = self._lg_side_mean(family, side, lg_all)
                fano = self._fano(family, side)
                home_adj = lg_side / lg_all if lg_all > 0 else 1.0

                att = self._shrunk_rate(tstate.dec(tstate.for_, family), day, lg_all)
                con = self._shrunk_rate(ostate.dec(ostate.conc, family), day, lg_all)

                # A — aditivo: promedio de lo que genero y lo que concede el rival.
                mean_a = max(0.5 * (att + con) * home_adj, 1e-6)
                out["A"][col_key(family, side)] = (mean_a, mean_a * fano)

                # B — multiplicativo (matchup).
                mean_b = max(att * con / lg_all * home_adj if lg_all > 0 else att, 1e-6)
                out["B"][col_key(family, side)] = (mean_b, mean_b * fano)

                # bobo — media/var de la liga-TEMPORADA hasta hoy (fallback all-time).
                mom_s = self._mom_season(season, family, side)
                if mom_s.n >= 30 and mom_s.mean is not None:
                    mean_d = mom_s.mean
                    var_d = mom_s.var or (mean_d * fano)
                else:
                    mean_d = lg_side
                    var_d = mean_d * fano
                out["bobo"][col_key(family, side)] = (max(mean_d, 1e-6), max(var_d, 1e-9))

                # C — GLM: features point-in-time (se guardan para entrenar al revelar).
                x = self._features_c(
                    family=family, side=side, team=team, opp=opp,
                    tstate=tstate, day=day, att=att, con=con, lg_all=lg_all,
                )
                beta = self._c_beta[family]
                if beta is not None:
                    mu = float(np.exp(np.clip(x @ beta, -12.0, 6.0)))
                    mean_c = min(max(mu, 1e-6), lg_all * 6 + 5)
                else:
                    mean_c = mean_a  # sin modelo aún: hereda A (se flaggea en el reporte)
                out["C"][col_key(family, side)] = (mean_c, mean_c * fano)

                actual = float(row[col_of(family, side)])
                self._pending.append((family, side, x, actual))

        return out

    def _features_c(
        self, *, family: str, side: str, team: str, opp: str,
        tstate: _TeamState, day: pd.Timestamp, att: float, con: float, lg_all: float,
    ) -> np.ndarray:
        cfg = self.cfg
        rest = 7.0
        if tstate.last_match is not None:
            rest = float(min(max((day - tstate.last_match).days, 1), 21))
        form_dec = tstate.dec(tstate.form, family)
        form = form_dec.rate(day, cfg.form_halflife_days, lg_all)
        h2h_dec = self.h2h.get((team, opp), {}).get(family)
        if h2h_dec is not None and h2h_dec.eff_n(day, cfg.halflife_days * 2) >= 2.0:
            h2h = h2h_dec.rate(day, cfg.halflife_days * 2, lg_all)
        else:
            h2h = lg_all
        eps = 0.25
        return np.array([
            1.0,
            1.0 if side == "h" else 0.0,
            np.log((att + eps) / (lg_all + eps)),
            np.log((con + eps) / (lg_all + eps)),
            np.log(rest / 7.0),
            np.log((form + eps) / (lg_all + eps)),
            np.log((h2h + eps) / (lg_all + eps)),
        ])

    # ---------- revelado (después de predecir TODO el día) ----------

    def reveal(self, row: pd.Series, day: pd.Timestamp) -> None:
        """Incorpora el resultado del partido al estado (tasas, liga, H2H, filas C)."""
        cfg = self.cfg
        home, away, season = row["home_team"], row["away_team"], row["season"]
        th, ta = self._team(home), self._team(away)
        for family in FAMILIES:
            vh = float(row[col_of(family, "h")])
            va = float(row[col_of(family, "a")])
            th.dec(th.for_, family).update(vh, day, cfg.halflife_days)
            th.dec(th.conc, family).update(va, day, cfg.halflife_days)
            ta.dec(ta.for_, family).update(va, day, cfg.halflife_days)
            ta.dec(ta.conc, family).update(vh, day, cfg.halflife_days)
            th.dec(th.form, family).update(vh, day, cfg.form_halflife_days)
            ta.dec(ta.form, family).update(va, day, cfg.form_halflife_days)
            self._mom(family, "h").add(vh)
            self._mom(family, "a").add(va)
            self._mom_season(season, family, "h").add(vh)
            self._mom_season(season, family, "a").add(va)
            self.h2h.setdefault((home, away), {}).setdefault(family, _Decayed()).update(
                vh, day, cfg.halflife_days * 2
            )
            self.h2h.setdefault((away, home), {}).setdefault(family, _Decayed()).update(
                va, day, cfg.halflife_days * 2
            )
        th.last_match = day
        ta.last_match = day

    def end_day(self, day: pd.Timestamp) -> None:
        """Cierra el día: vuelca las filas de entrenamiento del C y refitea si toca.

        Se llama DESPUÉS de predecir y revelar todos los partidos del día — así las
        features capturadas al predecir entran al set de entrenamiento recién cuando
        sus resultados ya son pasado (replay point-in-time exacto).
        """
        for family, _side, x, y in self._pending:
            self._c_x[family].append(x)
            self._c_y[family].append(y)
        self._pending.clear()
        self.maybe_refit_c(day)

    def maybe_refit_c(self, day: pd.Timestamp) -> None:
        """Refit periódico del GLM del Cerebro C usando SOLO filas ya reveladas."""
        cfg = self.cfg
        if self._c_last_fit is not None and (day - self._c_last_fit).days < cfg.refit_days:
            return
        fitted = False
        for family in FAMILIES:
            y = self._c_y[family]
            if len(y) < cfg.min_fit_rows:
                continue
            x = np.vstack(self._c_x[family])
            beta = _fit_poisson_irls(x, np.asarray(y, dtype=float), l2=cfg.l2)
            if beta is not None:
                self._c_beta[family] = beta
                fitted = True
        if fitted or self._c_last_fit is None:
            self._c_last_fit = day


def col_key(family: str, side: str) -> str:
    """Nombre de la cantidad predicha, ej ('corners','h') → 'corners_h'."""
    return f"{family}_{side}"


# Enganche para la Fase B (props): un ajustador opcional que recibe
# (home, away, {cantidad: (media, var)}) y devuelve el dict ajustado (ej. por
# bajas/XI confirmado). Por ahora NADIE lo implementa — es el punto de extensión.
PlayerAdjuster = (
    "Callable[[str, str, dict[str, tuple[float, float]]],"
    " dict[str, tuple[float, float]]]"
)
