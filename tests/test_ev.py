"""Tests del motor de EV y detección de value (Agente 3)."""

from __future__ import annotations

import pytest

from mundial_bot.value.ev import (
    Selection,
    evaluate,
    expected_value,
    find_value_bets,
)


def _sel(odds: float, selection: str = "home") -> Selection:
    return Selection(
        match="Argentina vs Mexico", market="1X2", selection=selection,
        odds=odds, bookmaker="Bet365",
    )


def test_expected_value_positivo_cuando_modelo_supera_implicita():
    # Modelo: 55%; cuota 2.00 implica 50% → +EV.
    assert expected_value(0.55, 2.00) == pytest.approx(0.10)


def test_expected_value_negativo_cuando_modelo_es_menor():
    assert expected_value(0.45, 2.00) == pytest.approx(-0.10)


def test_expected_value_cero_en_el_punto_justo():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)


def test_evaluate_marca_value_y_calcula_edge():
    pick = evaluate(_sel(2.00), model_prob=0.55, fair_prob=0.50)

    assert pick.is_value
    assert pick.edge == pytest.approx(0.10)
    assert pick.ev == pytest.approx(0.10)
    assert pick.model_vs_market == pytest.approx(0.05)


def test_evaluate_no_value_cuando_edge_negativo():
    pick = evaluate(_sel(2.00), model_prob=0.45)
    assert not pick.is_value
    assert pick.model_vs_market is None  # sin fair_prob no se puede comparar


def test_find_value_bets_filtra_por_umbral_y_ordena():
    candidates = [
        (_sel(2.00, "home"), 0.55),   # edge +0.10  → entra
        (_sel(3.50, "draw"), 0.30),   # edge +0.05  → entra
        (_sel(4.00, "away"), 0.20),   # edge -0.20  → afuera
        (_sel(2.00, "home"), 0.51),   # edge +0.02  → afuera (< 0.03)
    ]

    picks = find_value_bets(candidates, min_edge=0.03)

    assert len(picks) == 2
    # Ordenado de mayor a menor edge.
    assert picks[0].edge > picks[1].edge
    assert picks[0].selection.selection == "home"


def test_expected_value_valida_inputs():
    with pytest.raises(ValueError):
        expected_value(1.5, 2.0)   # prob fuera de [0,1]
    with pytest.raises(ValueError):
        expected_value(0.5, 0.9)   # cuota <= 1
