"""Tests de los modelos de córners y tarjetas (Fase 8)."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.count_market import closest_line, over_under


def _events() -> pd.DataFrame:
    """Eventos sintéticos: A genera más córners y recibe más tarjetas que B."""
    rows = []
    for i in range(10):
        rows.append({"match_id": i, "team": "A", "opponent": "B",
                     "corners_for": 6, "corners_against": 4, "cards": 2, "fouls": 12,
                     "referee": "RefDuro"})
        rows.append({"match_id": i, "team": "B", "opponent": "A",
                     "corners_for": 4, "corners_against": 6, "cards": 1, "fouls": 10,
                     "referee": "RefDuro"})
    return pd.DataFrame(rows)


# ---------- count_market ----------

def test_over_under_suma_uno():
    p_over, p_under = over_under(10.0, 9.5)
    assert p_over + p_under == pytest.approx(1.0)
    assert 0 < p_over < 1


def test_over_under_mas_probable_over_si_esperado_alto():
    p_over, _ = over_under(12.0, 9.5)   # esperado 12 >> línea 9.5
    assert p_over > 0.5


def test_closest_line():
    assert closest_line(10.3, (8.5, 9.5, 10.5, 11.5)) == 10.5


def test_over_under_sin_sobredispersion_es_poisson():
    # variance <= mean → cae a Poisson (mismo resultado que sin variance).
    assert over_under(10.0, 9.5, variance=8.0) == over_under(10.0, 9.5)


def test_negative_binomial_engorda_las_colas():
    # Con sobre-dispersión, una línea bien por encima de la media es más probable
    # que bajo Poisson (cola más gorda).
    p_over_nb, _ = over_under(10.0, 14.5, variance=25.0)
    p_over_poisson, _ = over_under(10.0, 14.5)
    assert p_over_nb > p_over_poisson
    assert 0 < p_over_nb < 1


# ---------- córners ----------

def test_corners_model_predice_total_coherente():
    model = CornersModel.from_events(_events())
    pred = model.predict("A", "B")

    # A_for(6)*B_against(6)/5 + B_for(4)*A_against(4)/5 = 7.2 + 3.2 = 10.4
    assert pred.total == pytest.approx(10.4, abs=0.1)
    assert pred.home_corners > pred.away_corners   # A genera más
    assert pred.p_over + pred.p_under == pytest.approx(1.0)


def test_corners_equipo_desconocido_usa_promedio_liga():
    model = CornersModel.from_events(_events())
    pred = model.predict("Narnia", "B")
    assert pred.total > 0   # no rompe, cae al promedio


# ---------- tarjetas ----------

def test_cards_model_usa_arbitro_y_equipos():
    model = CardsModel.from_events(_events())
    pred = model.predict("A", "B", referee="RefDuro")

    # team_base = 2+1 = 3 ; ref_base = 3 (total por partido) ; total = 3.0
    assert pred.total == pytest.approx(3.0, abs=0.01)
    assert pred.referee == "RefDuro"
    assert pred.p_over + pred.p_under == pytest.approx(1.0)


def test_cards_knockout_sube_el_total():
    model = CardsModel.from_events(_events())
    grupo = model.predict("A", "B", referee="RefDuro", knockout=False)
    eliminacion = model.predict("A", "B", referee="RefDuro", knockout=True)
    assert eliminacion.total > grupo.total


def test_cards_arbitro_desconocido_usa_promedio_liga():
    model = CardsModel.from_events(_events())
    pred = model.predict("A", "B", referee="RefФantasma")
    assert pred.total > 0   # cae al promedio de la liga
