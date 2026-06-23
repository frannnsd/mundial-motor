"""Tests del modelo de tiros al arco (shots on goal)."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.models.shots_model import ShotsModel


def _events() -> pd.DataFrame:
    """A patea más al arco que B (y recibe menos)."""
    rows = []
    for i in range(10):
        rows.append({"match_id": i, "team": "A", "opponent": "B",
                     "sot_for": 7, "sot_against": 3, "is_home": 1})
        rows.append({"match_id": i, "team": "B", "opponent": "A",
                     "sot_for": 3, "sot_against": 7, "is_home": 0})
    return pd.DataFrame(rows)


def test_shots_model_predice_total_y_over_under():
    m = ShotsModel.from_events(_events())
    assert m is not None
    pred = m.predict("A", "B")
    assert pred.total > 0
    assert pred.p_over + pred.p_under == pytest.approx(1.0)
    assert pred.home_shots > pred.away_shots      # A patea más


def test_shots_model_sin_columnas_devuelve_none():
    df = pd.DataFrame([{"match_id": 1, "team": "A", "opponent": "B", "corners_for": 5}])
    assert ShotsModel.from_events(df) is None


def test_shots_model_equipo_desconocido_usa_promedio_liga():
    m = ShotsModel.from_events(_events())
    pred = m.predict("Marte", "Venus")             # nadie conocido → media de liga
    assert pred.total > 0
