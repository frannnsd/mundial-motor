"""Tests del modelo Elo internacional (Agente 2 — base)."""

from __future__ import annotations

import pandas as pd

from mundial_bot.models.elo_model import (
    EloModel,
    goal_diff_multiplier,
    tournament_k,
)


def test_tournament_k_pondera_por_importancia():
    # La final del Mundial pesa más que un amistoso.
    assert tournament_k("FIFA World Cup") == 60
    assert tournament_k("Friendly") == 20
    assert tournament_k("FIFA World Cup qualification") == 40
    assert tournament_k("UEFA Euro") == 50
    # Torneo desconocido → default intermedio.
    assert tournament_k("Some Minor Cup") == 30


def test_goal_diff_multiplier_crece_con_la_goleada():
    assert goal_diff_multiplier(1) == 1.0       # diferencia de 1 o empate
    assert goal_diff_multiplier(2) == 1.5       # diferencia de 2
    assert goal_diff_multiplier(3) > 1.5        # 3+ crece
    assert goal_diff_multiplier(4) > goal_diff_multiplier(3)


def test_expected_score_simetrico_con_ratings_iguales_en_neutral():
    model = EloModel()
    # Mismo rating, cancha neutral → expectativa 0.5 para ambos.
    assert abs(model.expected_score("A", "B", neutral=True) - 0.5) < 1e-9


def test_localia_sube_la_expectativa():
    model = EloModel()
    neutral = model.expected_score("A", "B", neutral=True)
    en_casa = model.expected_score("A", "B", neutral=False)
    assert en_casa > neutral  # la ventaja de localía favorece al local


def test_predict_devuelve_probabilidades_que_suman_uno():
    model = EloModel()
    model.ratings["Brazil"] = 2100
    model.ratings["Bolivia"] = 1500

    p = model.predict("Brazil", "Bolivia", neutral=True)

    assert abs(p.home + p.draw + p.away - 1.0) < 1e-9
    assert p.home > p.away  # el favorito tiene más prob. de ganar
    assert 0 <= p.draw <= 1


def test_update_es_suma_cero_y_premia_al_ganador():
    model = EloModel()
    model.ratings["A"] = 1500
    model.ratings["B"] = 1500
    before = model.rating("A") + model.rating("B")

    model.update("A", "B", home_score=2, away_score=0, tournament="Friendly", neutral=True)

    after = model.rating("A") + model.rating("B")
    assert abs(before - after) < 1e-6   # suma cero
    assert model.rating("A") > 1500     # el ganador sube
    assert model.rating("B") < 1500     # el perdedor baja


def test_fit_sobre_dataframe_genera_rankings():
    # Arrange: tres partidos donde A le gana a todos.
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "B"],
            "home_score": [3, 2, 1],
            "away_score": [0, 0, 0],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [True, True, True],
        }
    )

    model = EloModel().fit(df)
    rankings = model.rankings()

    # Assert: A queda primero por ganar siempre.
    assert rankings[0][0] == "A"
    assert model.rating("A") > model.rating("B")
