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

from dataclasses import dataclass, field

import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights

DEFAULT_XI = 0.0018  # decaimiento temporal (half-life ~1 año); calibrable en Fase 5


class GoalsModelError(RuntimeError):
    """El modelo de goles no pudo predecir este partido (cae a Elo)."""


def _val(x):
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

    def fit(self, df: pd.DataFrame) -> GoalsModel:
        """Entrena el modelo. df: date, home_team, away_team, home_score, away_score, neutral."""
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
        """Predice los mercados de un partido. Eleva GoalsModelError si no puede."""
        if self._model is None:
            raise GoalsModelError("El modelo de goles no fue entrenado.")
        if not self.can_predict(home, away):
            raise GoalsModelError(f"Equipo sin datos de entrenamiento: {home} o {away}.")
        try:
            grid = self._model.predict(home, away, neutral_venue=neutral)
            hda = grid.home_draw_away
            lines = {
                strike: (
                    float(grid.total_goals("over", strike)),
                    float(grid.total_goals("under", strike)),
                )
                for strike in (0.5, 1.5, 2.5, 3.5, 4.5)
            }
            return MatchMarkets(
                home=float(hda[0]),
                draw=float(hda[1]),
                away=float(hda[2]),
                over_2_5=lines[2.5][0],
                under_2_5=lines[2.5][1],
                btts_yes=_val(grid.btts_yes),
                btts_no=_val(grid.btts_no),
                home_xg=_val(grid.home_goal_expectation),
                away_xg=_val(grid.away_goal_expectation),
                lines=lines,
            )
        except (ValueError, KeyError) as e:
            raise GoalsModelError(f"Dixon-Coles falló para {home} vs {away}: {e}") from e
