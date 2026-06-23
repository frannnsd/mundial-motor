"""Tests del parseo de tiros por jugador (player props)."""

from __future__ import annotations

import math

import pytest

from mundial_bot.collectors.player_stats import (
    SquadGoals,
    _poisson_over_under,
    goalscorer_probs,
    match_casa_odd,
    opponent_factor,
    parse_player_shots,
    player_sot_casa_odds,
    player_sot_probs,
)


class _FakeShots:
    league_avg = 4.0
    team_against = {"Floja": 6.0, "Firme": 2.0, "Media": 4.0}

RAW = {
    "response": [
        {
            "player": {"name": "Lionel Messi"},
            "statistics": [
                {"team": {"name": "Argentina"}, "games": {"appearences": 5},
                 "shots": {"total": 20, "on": 10}},
                {"team": {"name": "Inter Miami"}, "games": {"appearences": 15},
                 "shots": {"total": 40, "on": 20}},
            ],
        }
    ]
}


def test_parse_suma_tiros_de_toda_la_temporada():
    ps = parse_player_shots(RAW, "Messi")
    assert ps is not None
    assert ps.appearances == 20            # 5 + 15
    assert ps.shots_total == 60 and ps.shots_on == 30
    assert ps.shots_per_game == pytest.approx(3.0)
    assert ps.sot_per_game == pytest.approx(1.5)


def test_parse_sin_partidos_devuelve_none():
    raw = {"response": [{"player": {"name": "X"},
                         "statistics": [{"games": {"appearences": 0}, "shots": {}}]}]}
    assert parse_player_shots(raw, "X") is None


def test_parse_respuesta_vacia():
    assert parse_player_shots({"response": []}, "Nadie") is None


def test_poisson_over_under_coherente():
    out = dict(_poisson_over_under(1.5, (0.5, 1.5, 2.5)))
    # Más de 0.5 = 1 - P(0) = 1 - e^-1.5
    assert out[0.5] == pytest.approx(1 - math.exp(-1.5))
    # Monótono decreciente al subir la línea
    assert out[0.5] > out[1.5] > out[2.5]


def test_goalscorer_reparte_el_xg_y_ordena_por_chance():
    squad = [
        SquadGoals("Goleador", appearances=10, goals=8),    # 0.8/PJ
        SquadGoals("Media", appearances=10, goals=2),        # 0.2/PJ
        SquadGoals("Defensor", appearances=10, goals=0),     # no convierte → excluido
        SquadGoals("Suplente", appearances=1, goals=1),      # pocas apariciones → excluido
    ]
    rows = goalscorer_probs(squad, team_xg=2.0)
    names = [r[0] for r in rows]
    assert names == ["Goleador", "Media"]                    # ordenado por P(1+)
    # El goleador concentra 0.8/(0.8+0.2)=80% del xG del equipo → xg 1.6
    assert rows[0][1] == pytest.approx(1.6)
    assert rows[0][2] > rows[1][2]                           # más chance de 1+


def test_goalscorer_sin_goleadores_o_sin_xg():
    assert goalscorer_probs([SquadGoals("X", 5, 0)], 2.0) == []
    assert goalscorer_probs([SquadGoals("X", 5, 3)], 0.0) == []


def test_player_sot_probs_por_tasa_y_factor():
    squad = [
        SquadGoals("Tirador", appearances=10, goals=2, shots_on=15),   # 1.5 TA/PJ
        SquadGoals("Flojo", appearances=10, goals=0, shots_on=2),       # 0.2 TA/PJ
        SquadGoals("Nada", appearances=10, goals=0, shots_on=0),        # excluido
    ]
    rows = player_sot_probs(squad, factor=1.0)
    assert [r[0] for r in rows] == ["Tirador", "Flojo"]
    assert rows[0][2] > rows[1][2]            # más chance de 1+
    # Factor del rival escala la tasa → sube la prob.
    rows_floja = player_sot_probs(squad, factor=1.3)
    assert rows_floja[0][2] > rows[0][2]


def test_player_sot_casa_odds_y_match_por_apellido():
    best = {
        "Lionel Messi - 1+": (1.40, "Bet365"),
        "Julian Alvarez - 1+": (1.85, "Bet365"),
        "Algo raro - 2+": (5.0, "Bet365"),   # 2+ se ignora
    }
    casa = player_sot_casa_odds(best)
    assert "2+" not in str(casa)              # solo 1+
    assert match_casa_odd("L. Messi", casa) == (1.40, "Bet365")     # por apellido
    assert match_casa_odd("Julián Álvarez", casa) == (1.85, "Bet365")  # acentos
    assert match_casa_odd("Nadie", casa) is None


def test_opponent_factor_ajusta_por_la_defensa_del_rival():
    s = _FakeShots()
    assert opponent_factor(s, "Media") == pytest.approx(1.0)     # concede la media
    assert opponent_factor(s, "Floja") == pytest.approx(1.5)     # 6/4 = patea más
    assert opponent_factor(s, "Firme") == pytest.approx(0.6)     # 2/4=0.5 → acotado a 0.6
    assert opponent_factor(s, "Desconocido") == pytest.approx(1.0)  # sin dato → media
