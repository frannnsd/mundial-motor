"""Tests de las máscaras de combinada conjunta del mismo partido."""

from __future__ import annotations

import numpy as np
import pytest

from mundial_bot.models.joint import _goals_mask


def _grids(n: int = 5):
    idx = np.arange(n)
    i = idx.reshape(-1, 1)
    j = idx.reshape(1, -1)
    return i, j, i - j, i + j


def _m(leg: dict):
    i, j, margin, total = _grids()
    return np.broadcast_to(_goals_mask(leg, i, j, margin, total), (5, 5))


def test_ganador_local():
    mask = _m({"market": "ganador", "side": "home"})
    assert mask[2, 1] and not mask[1, 2] and not mask[1, 1]


def test_goles_under_y_over():
    under = _m({"market": "goles", "side": "under", "line": 2.5})
    assert under[1, 0] and not under[2, 1]          # total 1 sí, total 3 no
    over = _m({"market": "goles", "side": "over", "line": 2.5})
    assert over[2, 1] and not over[1, 0]


def test_ambos_marcan():
    yes = _m({"market": "ambos_marcan", "side": "yes"})
    assert yes[1, 1] and not yes[2, 0]


def test_handicap_local_menos_1_5():
    mask = _m({"market": "handicap", "team": "home", "line": -1.5})
    assert mask[2, 0] and not mask[1, 0]            # cubre si gana por 2+


def test_total_equipo_local():
    mask = _m({"market": "total_equipo", "team": "home", "side": "over", "line": 1.5})
    assert mask[2, 0] and not mask[1, 4]            # local marca 2+ goles


def test_linea_entera_rechazada_por_el_push():
    # Líneas enteras tienen push (devolución) → no se modelan en combinadas: deben fallar.
    for leg in (
        {"market": "goles", "side": "over", "line": 2},
        {"market": "handicap", "team": "home", "line": -1},
        {"market": "total_equipo", "team": "away", "side": "under", "line": 1},
    ):
        with pytest.raises(ValueError, match="push"):
            _m(leg)
