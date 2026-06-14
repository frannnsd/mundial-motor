"""Tests del parseo de eventos de StatsBomb (Fase 8a)."""

from __future__ import annotations

import pandas as pd

from mundial_bot.collectors.statsbomb_stats import _counts_by_team


def test_counts_by_team_cuenta_corners_faltas_y_tarjetas():
    # Arrange: A → 2 córners, 1 falta, 1 tarjeta (por falta).
    #          B → 1 córner, 1 falta, 1 tarjeta (mala conducta).
    events = pd.DataFrame({
        "team": ["A", "A", "B", "A", "B", "B"],
        "type": ["Pass", "Pass", "Pass", "Foul Committed", "Foul Committed", "Bad Behaviour"],
        "pass_type": ["Corner", "Corner", "Corner", None, None, None],
        "foul_committed_card": [None, None, None, "Yellow Card", None, None],
        "bad_behaviour_card": [None, None, None, None, None, "Yellow Card"],
    })

    # Act
    counts = _counts_by_team(events)

    # Assert
    assert counts["A"]["corners"] == 2
    assert counts["B"]["corners"] == 1
    assert counts["A"]["fouls"] == 1
    assert counts["B"]["fouls"] == 1
    assert counts["A"]["cards"] == 1   # tarjeta por falta
    assert counts["B"]["cards"] == 1   # tarjeta por mala conducta


def test_counts_by_team_columnas_faltantes_no_rompe():
    # Si faltan columnas (partido sin datos), no debe explotar.
    events = pd.DataFrame({"team": ["A", "B"]})
    counts = _counts_by_team(events)
    assert isinstance(counts, dict)
