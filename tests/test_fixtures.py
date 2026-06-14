"""Tests del colector de fixtures de API-Football (Agente 1)."""

from __future__ import annotations

from mundial_bot.collectors.fixtures import Fixture, parse_fixtures

SAMPLE = {
    "response": [
        {
            "fixture": {"id": 1, "date": "2026-06-14T18:00:00+00:00", "referee": "Raphael Claus"},
            "league": {"id": 1, "name": "World Cup", "round": "Group Stage - 2"},
            "teams": {"home": {"name": "Morocco"}, "away": {"name": "Brazil"}},
        },
        {
            "fixture": {"id": 2, "date": "2026-06-14T21:00:00+00:00", "referee": None},
            "league": {"id": 1, "name": "World Cup", "round": "Round of 16"},
            "teams": {"home": {"name": "Argentina"}, "away": {"name": "France"}},
        },
    ]
}


def test_parse_fixtures_extrae_equipos_arbitro_y_ronda():
    fixtures = parse_fixtures(SAMPLE)

    assert len(fixtures) == 2
    assert fixtures[0].match == "Morocco vs Brazil"
    assert fixtures[0].referee == "Raphael Claus"
    assert fixtures[0].knockout is False   # fase de grupos
    assert fixtures[1].knockout is True    # octavos = eliminación


def test_parse_fixtures_saltea_partidos_sin_equipos():
    raw = {
        "response": [{"fixture": {"id": 9}, "league": {}, "teams": {"home": None, "away": None}}]
    }
    assert parse_fixtures(raw) == []


def test_parse_fixtures_respuesta_vacia():
    assert parse_fixtures({"response": []}) == []
    assert parse_fixtures({}) == []


def test_fixture_knockout_grupos_es_false():
    f = Fixture(home_team="A", away_team="B", date="", round="Group Stage - 1")
    assert f.knockout is False
