"""Tests del backtest de córners/tarjetas."""

from __future__ import annotations

import pandas as pd

from mundial_bot.backtest.count_backtest import _per_match_table, backtest_corners


def _dated_df() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=60, freq="3D")
    for i, d in enumerate(dates):
        rows.append({"match_id": i, "date": d, "team": "A", "opponent": "B",
                     "corners_for": 8, "corners_against": 4, "cards": 2, "fouls": 10,
                     "referee": "R", "is_home": 1})
        rows.append({"match_id": i, "date": d, "team": "B", "opponent": "A",
                     "corners_for": 4, "corners_against": 8, "cards": 1, "fouls": 10,
                     "referee": "R", "is_home": 0})
    return pd.DataFrame(rows)


def test_per_match_table_arma_totales():
    table = _per_match_table(_dated_df())
    assert "corners_total" in table.columns
    assert (table["corners_total"] == 12).all()       # 8 + 4
    assert (table["cards_total"] == 3).all()           # 2 + 1
    assert list(table["date"]) == sorted(table["date"])  # ordenado por fecha


def test_backtest_corners_corre_y_reporta():
    res = backtest_corners(_dated_df(), start_frac=0.5, min_train=20)
    assert res.n > 0
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.naive_accuracy <= 1.0
    assert res.brier >= 0.0
    assert "córners" in res.summary()
