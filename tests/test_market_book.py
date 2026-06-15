"""Tests del libro de mercados: invariantes de probabilidad y cuota justa push-aware."""

from __future__ import annotations

import numpy as np
import pytest

from mundial_bot.models.market_book import _goals_selections, _sel

# Matriz simétrica simple (suma 1) para verificar a mano.
#        away0 away1 away2
# home0  0.2   0.1   0.0
# home1  0.1   0.2   0.1
# home2  0.0   0.1   0.2
MATRIX = np.array([[0.2, 0.1, 0.0], [0.1, 0.2, 0.1], [0.0, 0.1, 0.2]])


def _book(matrix=MATRIX):
    sels = _goals_selections(matrix, "Local", "Visita", 1.0, 1.0)
    return {(s.market, s.pick): s for s in sels}


def test_sel_clean_fair_is_inverse_prob():
    s = _sel("M", "x", 0.25)
    assert s.prob == pytest.approx(0.25)
    assert s.fair == pytest.approx(4.0)
    assert s.push == 0.0


def test_sel_push_aware_fair_and_effective_prob():
    # gana 0.2, push 0.6 → efectiva 0.5, justa (1-0.6)/0.2 = 2.0
    s = _sel("M", "x", 0.2, push=0.6)
    assert s.prob == pytest.approx(0.5)
    assert s.fair == pytest.approx(2.0)


def test_1x2_sums_to_one():
    b = _book()
    total = (
        b[("Ganador (1X2)", "Gana Local")].prob
        + b[("Ganador (1X2)", "Empate")].prob
        + b[("Ganador (1X2)", "Gana Visita")].prob
    )
    assert total == pytest.approx(1.0)
    assert b[("Ganador (1X2)", "Gana Local")].prob == pytest.approx(0.2)
    assert b[("Ganador (1X2)", "Empate")].prob == pytest.approx(0.6)


def test_double_chance_and_dnb_push():
    b = _book()
    assert b[("Doble oportunidad", "Local o empate")].prob == pytest.approx(0.8)
    dnb = b[("Empate no apuesta", "Local")]
    assert dnb.push == pytest.approx(0.6)       # empate = devolución
    assert dnb.prob == pytest.approx(0.5)        # 0.2 / (1 - 0.6)
    assert dnb.fair == pytest.approx(2.0)


def test_totals_over_under_complement():
    b = _book()
    over = b[("Goles Más/Menos", "Más de 1.5")].prob
    under = b[("Goles Más/Menos", "Menos de 1.5")].prob
    assert over == pytest.approx(0.6)
    assert over + under == pytest.approx(1.0)


def test_btts_complement():
    b = _book()
    yes = b[("Ambos marcan", "Sí")].prob
    no = b[("Ambos marcan", "No")].prob
    assert yes == pytest.approx(0.6)
    assert yes + no == pytest.approx(1.0)


def test_asian_handicap_half_line_equals_winner():
    b = _book()
    # Local -0.5 gana solo si gana el partido → = prob de ganar (0.2).
    assert b[("Hándicap asiático", "Local -0.5")].prob == pytest.approx(0.2)
    # Local +0.5 cubre si gana o empata → doble oportunidad (0.8).
    assert b[("Hándicap asiático", "Local +0.5")].prob == pytest.approx(0.8)


def test_asian_handicap_level_has_push():
    b = _book()
    level = b[("Hándicap asiático", "Local 0")]
    assert level.push == pytest.approx(0.6)      # empate devuelve
    assert level.prob == pytest.approx(0.5)       # 0.2 / 0.4
