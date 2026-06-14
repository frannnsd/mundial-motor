"""Tests del backtesting (métricas + walk-forward) — Agente 5."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.backtest.metrics import (
    brier_1x2,
    log_loss_1x2,
    outcome_index,
    rps_1x2,
)
from mundial_bot.backtest.walk_forward import walk_forward_elo

# ---------- Métricas ----------

def test_outcome_index():
    assert outcome_index(2, 0) == 0  # local
    assert outcome_index(1, 1) == 1  # empate
    assert outcome_index(0, 3) == 2  # visitante


def test_rps_perfecto_es_cero():
    assert rps_1x2([1.0, 0.0, 0.0], 0) == pytest.approx(0.0)


def test_rps_valor_conocido():
    # probs=[0.5,0.3,0.2], gana local (idx 0):
    # corte1 (0.5-1)^2=0.25 ; corte2 (0.8-1)^2=0.04 ; /2 = 0.145
    assert rps_1x2([0.5, 0.3, 0.2], 0) == pytest.approx(0.145)


def test_rps_penaliza_mas_el_error_lejano():
    # Predecir fuerte al local cuando gana el visitante es peor que predecir empate.
    probs = [0.7, 0.2, 0.1]
    assert rps_1x2(probs, 2) > rps_1x2(probs, 1)


def test_brier_y_logloss_perfectos():
    assert brier_1x2([1.0, 0.0, 0.0], 0) == pytest.approx(0.0)
    assert log_loss_1x2([1.0, 0.0, 0.0], 0) == pytest.approx(0.0, abs=1e-9)


# ---------- Walk-forward ----------

def test_walk_forward_produce_metricas_validas():
    # Dataset: A casi siempre gana → modelo debería aprender y acertar > azar.
    rows = []
    dates = pd.date_range("2015-01-01", periods=120, freq="7D")
    teams_away = ["B", "C", "D"]
    for i, d in enumerate(dates):
        away = teams_away[i % 3]
        rows.append({
            "date": d, "home_team": "A", "away_team": away,
            "home_score": 2, "away_score": 0,
            "tournament": "Friendly", "neutral": True,
        })
    df = pd.DataFrame(rows)

    result = walk_forward_elo(df, start="2016-06-01")

    assert result.n > 0
    assert 0.0 <= result.rps <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    # A siempre gana de local → tras calentar, debería acertar la mayoría.
    assert result.accuracy > 0.5


def test_walk_forward_sin_partidos_falla():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2015-01-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [1], "away_score": [0],
        "tournament": ["Friendly"], "neutral": [True],
    })
    with pytest.raises(ValueError, match="No hubo partidos"):
        walk_forward_elo(df, start="2030-01-01")
