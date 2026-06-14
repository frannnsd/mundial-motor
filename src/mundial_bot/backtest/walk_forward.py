"""Backtest walk-forward del modelo Elo (validación out-of-sample real).

Recorre los partidos en orden cronológico. Para cada partido a partir de
``start``: predice con los ratings ENTRENADOS SOLO CON EL PASADO, registra las
métricas contra el resultado real, y RECIÉN DESPUÉS actualiza los ratings. Así no
hay leakage: nunca se predice con información del futuro.

Los partidos previos a ``start`` solo se usan para "calentar" los ratings (sin
puntuar), porque al inicio todos arrancan en 1500 y esas predicciones serían ruido.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mundial_bot.backtest.metrics import (
    brier_1x2,
    log_loss_1x2,
    outcome_index,
    rps_1x2,
)
from mundial_bot.models.elo_model import EloConfig, EloModel


@dataclass(frozen=True)
class BacktestResult:
    n: int
    rps: float
    brier: float
    log_loss: float
    accuracy: float

    def summary(self) -> str:
        return (
            f"Partidos evaluados : {self.n:,}\n"
            f"RPS                : {self.rps:.4f}  (mejor cuanto más bajo; ~0.17-0.19 es bueno)\n"
            f"Brier              : {self.brier:.4f}\n"
            f"Log-loss           : {self.log_loss:.4f}\n"
            f"Accuracy (1X2)     : {self.accuracy:.1%}"
        )


def walk_forward_elo(
    df: pd.DataFrame,
    *,
    start: str = "2018-01-01",
    config: EloConfig | None = None,
) -> BacktestResult:
    """Corre el backtest walk-forward del Elo y devuelve las métricas agregadas."""
    model = EloModel(config or EloConfig())
    start_ts = pd.Timestamp(start)

    n = 0
    rps_sum = brier_sum = ll_sum = correct = 0.0

    for row in df.sort_values("date").itertuples(index=False):
        if row.date >= start_ts:
            p = model.predict(row.home_team, row.away_team, neutral=bool(row.neutral))
            probs = [p.home, p.draw, p.away]
            idx = outcome_index(int(row.home_score), int(row.away_score))

            rps_sum += rps_1x2(probs, idx)
            brier_sum += brier_1x2(probs, idx)
            ll_sum += log_loss_1x2(probs, idx)
            if max(range(3), key=lambda i: probs[i]) == idx:
                correct += 1
            n += 1

        model.update(
            row.home_team,
            row.away_team,
            home_score=int(row.home_score),
            away_score=int(row.away_score),
            tournament=getattr(row, "tournament", "Friendly"),
            neutral=bool(row.neutral),
        )

    if n == 0:
        raise ValueError("No hubo partidos para evaluar (revisá la fecha 'start').")

    return BacktestResult(
        n=n,
        rps=rps_sum / n,
        brier=brier_sum / n,
        log_loss=ll_sum / n,
        accuracy=correct / n,
    )
