"""Tests del parseo de tiros por jugador (player props)."""

from __future__ import annotations

import math

import pytest

from mundial_bot.collectors.player_stats import (
    _poisson_over_under,
    opponent_factor,
    parse_player_shots,
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


def test_opponent_factor_ajusta_por_la_defensa_del_rival():
    s = _FakeShots()
    assert opponent_factor(s, "Media") == pytest.approx(1.0)     # concede la media
    assert opponent_factor(s, "Floja") == pytest.approx(1.5)     # 6/4 = patea más
    assert opponent_factor(s, "Firme") == pytest.approx(0.6)     # 2/4=0.5 → acotado a 0.6
    assert opponent_factor(s, "Desconocido") == pytest.approx(1.0)  # sin dato → media
