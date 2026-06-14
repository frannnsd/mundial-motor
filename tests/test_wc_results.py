"""Tests del colector de resultados del Mundial (autoalimentación del Elo)."""

from __future__ import annotations

from mundial_bot.collectors.wc_results import parse_wc_results

SAMPLE = {
    "response": [
        {
            "fixture": {"date": "2026-06-14T18:00:00+00:00", "status": {"short": "FT"}},
            "goals": {"home": 2, "away": 1},
            "teams": {"home": {"name": "Argentina"}, "away": {"name": "Mexico"}},
        },
        {
            "fixture": {"date": "2026-06-15T18:00:00+00:00", "status": {"short": "NS"}},
            "goals": {"home": None, "away": None},
            "teams": {"home": {"name": "Spain"}, "away": {"name": "Brazil"}},
        },
    ]
}


def test_parse_wc_results_solo_partidos_terminados():
    df = parse_wc_results(SAMPLE)

    # Solo el partido FT entra (el NS no).
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team"] == "Argentina"
    assert row["away_team"] == "Mexico"
    assert int(row["home_score"]) == 2
    assert int(row["away_score"]) == 1
    assert row["tournament"] == "FIFA World Cup"
    assert bool(row["neutral"]) is True


def test_parse_wc_results_vacio():
    assert parse_wc_results({"response": []}).empty
    assert parse_wc_results({}).empty
