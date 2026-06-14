"""Tests del colector de lesiones/suspensiones."""

from __future__ import annotations

from mundial_bot.collectors.injuries import injury_counts, parse_injuries

SAMPLE = {
    "response": [
        {"team": {"name": "Argentina"},
         "player": {"name": "Messi", "reason": "Knock", "type": "Missing Fixture"}},
        {"team": {"name": "Argentina"},
         "player": {"name": "Otamendi", "reason": "Suspended", "type": "Missing Fixture"}},
        {"team": {"name": "Brazil"},
         "player": {"name": "Neymar", "reason": "Injury", "type": "Missing Fixture"}},
    ]
}


def test_parse_injuries_agrupa_por_equipo():
    injuries = parse_injuries(SAMPLE)
    assert set(injuries) == {"Argentina", "Brazil"}
    assert len(injuries["Argentina"]) == 2
    assert injuries["Brazil"][0].player == "Neymar"


def test_injury_counts():
    counts = injury_counts(parse_injuries(SAMPLE))
    assert counts["Argentina"] == 2
    assert counts["Brazil"] == 1


def test_parse_injuries_ignora_incompletos():
    raw = {"response": [{"team": {"name": "X"}, "player": {}}]}
    assert parse_injuries(raw) == {}
