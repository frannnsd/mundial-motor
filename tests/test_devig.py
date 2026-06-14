"""Tests del de-vig de cuotas (Agente 3)."""

from __future__ import annotations

import pytest

from mundial_bot.value.devig import (
    FairProbabilities,
    decimal_to_implied,
    devig,
    overround,
)


def test_decimal_to_implied():
    assert decimal_to_implied(2.0) == 0.5
    assert decimal_to_implied(4.0) == 0.25


def test_decimal_to_implied_rechaza_cuota_invalida():
    with pytest.raises(ValueError):
        decimal_to_implied(0.9)


def test_overround_es_positivo_en_mercado_real():
    # Un mercado 1X2 real siempre tiene margen > 0.
    margin = overround({"home": 2.10, "draw": 3.40, "away": 3.80})
    assert margin > 0
    # 1/2.10 + 1/3.40 + 1/3.80 - 1 ≈ 0.0335
    assert margin == pytest.approx(0.0335, abs=0.005)


def test_devig_shin_probabilidades_suman_uno_y_quitan_margen():
    fair = devig({"home": 2.10, "draw": 3.40, "away": 3.80}, method="shin")

    assert isinstance(fair, FairProbabilities)
    assert sum(fair.outcomes.values()) == pytest.approx(1.0, abs=1e-9)
    assert fair.margin > 0
    # El favorito (cuota más baja) tiene la prob. justa más alta.
    assert fair["home"] > fair["away"]


def test_devig_mercado_dos_vias():
    fair = devig({"over": 1.90, "under": 1.90}, method="shin")
    # Mercado simétrico → ~50/50.
    assert fair["over"] == pytest.approx(0.5, abs=0.02)
    assert sum(fair.outcomes.values()) == pytest.approx(1.0, abs=1e-9)


def test_devig_rechaza_metodo_invalido():
    with pytest.raises(ValueError, match="no soportado"):
        devig({"home": 2.0, "away": 2.0}, method="inexistente")


def test_devig_rechaza_mercado_de_un_resultado():
    with pytest.raises(ValueError, match="al menos 2"):
        devig({"home": 2.0})
