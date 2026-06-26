"""Modelo de goles Dixon-Coles — Agente 2 (mercados de goles).

Envuelve `DixonColesGoalModel` de penaltyblog con dos cosas clave para selecciones:
  - **time-decay** (`xi`): los partidos recientes pesan más (peso exponencial).
  - **neutral_venue**: el Mundial se juega en cancha neutral (sin ventaja de localía).

El Elo es mejor baseline para 1X2 en datos ralos; Dixon-Coles brilla para los
mercados de **goles** (over/under, ambos marcan). Por eso de acá tomamos sobre todo
over/under y BTTS, y dejamos 1X2 como respaldo.

Robustez: con equipos muy disparejos o poca data, la corrección tau de Dixon-Coles
puede generar celdas negativas y penaltyblog lanza ValueError. Lo capturamos y
elevamos `GoalsModelError` para que el pipeline caiga a Elo en ese partido.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights
from scipy.stats import poisson

DEFAULT_XI = 0.0018  # decaimiento temporal (half-life ~1 año); calibrable en Fase 5


class GoalsModelError(RuntimeError):
    """El modelo de goles no pudo predecir este partido (cae a Elo)."""


def _val(x: float | Callable[[], float]) -> float:
    """Algunos atributos de la grilla son propiedades; otros, métodos. Normaliza."""
    return x() if callable(x) else float(x)


@dataclass(frozen=True)
class MatchMarkets:
    """Probabilidades de múltiples mercados de un partido (Dixon-Coles)."""

    home: float
    draw: float
    away: float
    over_2_5: float
    under_2_5: float
    btts_yes: float
    btts_no: float
    home_xg: float
    away_xg: float
    # Over/Under por línea de goles: {strike: (prob_over, prob_under)}
    lines: dict[float, tuple[float, float]] = field(default_factory=dict)

    def one_x_two(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}

    @property
    def exp_goals(self) -> float:
        return self.home_xg + self.away_xg


class GoalsModel:
    """Modelo Dixon-Coles entrenado sobre resultados internacionales."""

    def __init__(self, xi: float = DEFAULT_XI):
        self.xi = xi
        self._model: DixonColesGoalModel | None = None
        self._teams: set[str] = set()
        self.calibration = 1.0  # corrige el sesgo de goles (≠1 = el torneo va más/menos goleador)

    def _cal_matrix(
        self, matrix: np.ndarray, home_xg: float, away_xg: float
    ) -> tuple[np.ndarray, float, float]:
        """Aplica la calibración: reescala el xG y rearma la matriz si calibration ≠ 1."""
        if abs(self.calibration - 1.0) < 1e-6:
            return matrix, home_xg, away_xg
        lh, la = home_xg * self.calibration, away_xg * self.calibration
        k = np.arange(matrix.shape[0])
        pm = np.outer(poisson.pmf(k, max(lh, 1e-9)), poisson.pmf(k, max(la, 1e-9)))
        s = float(pm.sum())
        return (pm / s if s > 0 else pm), lh, la

    def fit_calibration(
        self, df: pd.DataFrame, *, min_matches: int = 15, bounds: tuple[float, float] = (0.9, 1.3)
    ) -> float:
        """Calibra el total de goles para matchear el ritmo real del df (ej. el Mundial).

        Corrige el sesgo del Dixon-Coles cuando el torneo tiene más (o menos) goles que el
        promedio histórico con el que se entrenó. Devuelve el factor aplicado.
        """
        needed = {"home_team", "away_team", "home_score", "away_score"}
        if df is None or df.empty or not needed.issubset(df.columns):
            return self.calibration
        self.calibration = 1.0  # medir el sesgo crudo
        pred: list[float] = []
        act: list[float] = []
        for _, r in df.iterrows():
            h, a = str(r["home_team"]), str(r["away_team"])
            if not self.can_predict(h, a):
                continue
            try:
                _, hx, ax = self.score_matrix(h, a, neutral=True)
            except GoalsModelError:
                continue
            pred.append(hx + ax)
            act.append(float(r["home_score"]) + float(r["away_score"]))
        if len(pred) >= min_matches and float(np.mean(pred)) > 0:
            f = float(np.mean(act)) / float(np.mean(pred))
            self.calibration = min(max(f, bounds[0]), bounds[1])
        return self.calibration

    REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}

    def fit(self, df: pd.DataFrame) -> GoalsModel:
        """Entrena el modelo. df: date, home_team, away_team, home_score, away_score, neutral."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise GoalsModelError(f"GoalsModel.fit() faltan columnas: {sorted(missing)}")
        weights = dixon_coles_weights(df["date"], xi=self.xi)
        self._model = DixonColesGoalModel(
            df["home_score"].to_numpy(),
            df["away_score"].to_numpy(),
            df["home_team"].to_numpy(),
            df["away_team"].to_numpy(),
            weights=weights,
            neutral_venue=df["neutral"].astype(int).to_numpy(),
        )
        self._model.fit()
        self._teams = set(df["home_team"]) | set(df["away_team"])
        return self

    @property
    def teams(self) -> set[str]:
        return self._teams

    def can_predict(self, home: str, away: str) -> bool:
        return home in self._teams and away in self._teams

    def predict(self, home: str, away: str, *, neutral: bool = False) -> MatchMarkets:
        """Predice los mercados de un partido desde la matriz (ya calibrada)."""
        matrix, hx, ax = self.score_matrix(home, away, neutral=neutral)
        n = matrix.shape[0]
        i = np.arange(n).reshape(-1, 1)
        j = np.arange(n).reshape(1, -1)
        margin = i - j
        total = i + j
        p_home = float(matrix[margin > 0].sum())
        p_draw = float(matrix[margin == 0].sum())
        p_away = float(matrix[margin < 0].sum())
        lines = {}
        for strike in (0.5, 1.5, 2.5, 3.5, 4.5):
            over = float(matrix[total > strike].sum())
            lines[strike] = (over, 1.0 - over)
        btts_yes = float(matrix[(i >= 1) & (j >= 1)].sum())
        return MatchMarkets(
            home=p_home, draw=p_draw, away=p_away,
            over_2_5=lines[2.5][0], under_2_5=lines[2.5][1],
            btts_yes=btts_yes, btts_no=1.0 - btts_yes,
            home_xg=hx, away_xg=ax, lines=lines,
        )

    def score_matrix(
        self, home: str, away: str, *, neutral: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """Matriz de marcadores P[i,j]=P(local i, visita j) + xG local/visita.

        De acá se derivan TODOS los mercados de goles (1X2, asiáticos, totales,
        ambos marcan, hándicaps, marcadores exactos…). Eleva GoalsModelError si no puede.
        """
        if self._model is None:
            raise GoalsModelError("El modelo de goles no fue entrenado.")
        if not self.can_predict(home, away):
            raise GoalsModelError(f"Equipo sin datos de entrenamiento: {home} o {away}.")
        try:
            grid = self._model.predict(home, away, neutral_venue=neutral)
            matrix = np.asarray(grid.grid, dtype=float)
            total = matrix.sum()
            if total <= 0:
                raise GoalsModelError(f"Matriz inválida para {home} vs {away}.")
            return self._cal_matrix(
                matrix / total,
                _val(grid.home_goal_expectation),
                _val(grid.away_goal_expectation),
            )
        except (ValueError, KeyError) as e:
            raise GoalsModelError(f"Dixon-Coles falló para {home} vs {away}: {e}") from e
