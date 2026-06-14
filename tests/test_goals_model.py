"""Tests del modelo de goles Dixon-Coles (Agente 2b)."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from mundial_bot.models.goals_model import GoalsModel, GoalsModelError, MatchMarkets


def _training_df() -> pd.DataFrame:
    """Dataset chico pero suficiente para un fit estable (round-robin repetido)."""
    teams = ["A", "B", "C", "D", "E"]
    scores = itertools.cycle([(2, 0), (1, 1), (0, 1), (3, 1), (1, 0), (2, 2), (0, 0)])
    rows = []
    dates = pd.date_range("2023-01-01", periods=200, freq="3D")
    di = iter(dates)
    for _ in range(5):
        for h, a in itertools.permutations(teams, 2):
            sh, sa = next(scores)
            rows.append(
                {
                    "date": next(di),
                    "home_team": h,
                    "away_team": a,
                    "home_score": sh,
                    "away_score": sa,
                    "neutral": True,
                }
            )
    return pd.DataFrame(rows)


def test_predict_devuelve_mercados_coherentes():
    model = GoalsModel().fit(_training_df())

    markets = model.predict("A", "B", neutral=True)

    assert isinstance(markets, MatchMarkets)
    # 1X2 suma ~1.
    assert markets.home + markets.draw + markets.away == pytest.approx(1.0, abs=1e-6)
    # over + under = 1 ; btts sí + no = 1.
    assert markets.over_2_5 + markets.under_2_5 == pytest.approx(1.0, abs=1e-6)
    assert markets.btts_yes + markets.btts_no == pytest.approx(1.0, abs=1e-6)
    # expectativas de gol positivas.
    assert markets.home_xg > 0
    assert markets.away_xg > 0


def test_predict_equipo_desconocido_eleva_error():
    model = GoalsModel().fit(_training_df())
    with pytest.raises(GoalsModelError, match="sin datos"):
        model.predict("A", "Narnia")


def test_predict_sin_entrenar_eleva_error():
    with pytest.raises(GoalsModelError, match="no fue entrenado"):
        GoalsModel().predict("A", "B")


def test_can_predict():
    model = GoalsModel().fit(_training_df())
    assert model.can_predict("A", "B")
    assert not model.can_predict("A", "Z")
