"""Tests del blend Elo+DC para 1X2."""

from __future__ import annotations

import pytest

from mundial_bot.models.blend import blend_1x2


def test_blend_pondera_y_normaliza():
    b = blend_1x2((0.8, 0.1, 0.1), (0.5, 0.3, 0.2), w_elo=0.5)
    assert sum(b) == pytest.approx(1.0)
    assert b[0] == pytest.approx(0.65)   # 0.5*0.8 + 0.5*0.5


def test_blend_w_elo_1_es_solo_elo():
    assert blend_1x2((0.6, 0.25, 0.15), (0.1, 0.1, 0.8), w_elo=1.0) == pytest.approx(
        (0.6, 0.25, 0.15)
    )


def test_blend_w_elo_0_es_solo_dc():
    assert blend_1x2((0.6, 0.25, 0.15), (0.2, 0.3, 0.5), w_elo=0.0) == pytest.approx(
        (0.2, 0.3, 0.5)
    )


def test_blend_acerca_el_inflado_de_elo_al_dc():
    # Elo infla al favorito (78%); DC lo ve 52% → el blend queda en el medio, más realista.
    b = blend_1x2((0.78, 0.12, 0.10), (0.52, 0.31, 0.17), w_elo=0.40)
    assert 0.55 < b[0] < 0.66
