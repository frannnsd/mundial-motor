"""Tests del cliente de The Odds API (Agente 3, parseo offline)."""

from __future__ import annotations

from pathlib import Path

from mundial_bot.value.odds import (
    best_1x2,
    best_book_for,
    load_sample,
    parse_events,
)

SAMPLE = Path(__file__).parent / "data" / "sample_odds.json"


def test_load_sample_parsea_dos_partidos():
    matches = load_sample(SAMPLE)
    assert len(matches) == 2
    assert matches[0].match == "Argentina vs Mexico"
    assert matches[0].home_team == "Argentina"


def test_best_1x2_toma_la_mejor_cuota_entre_casas():
    matches = load_sample(SAMPLE)
    best = best_1x2(matches[0])

    # Bet365: Arg 1.95 / Pinnacle: Arg 2.00 → mejor = 2.00
    assert best["home"] == 2.00
    # Draw: 3.4 vs 3.5 → 3.5 ; Away (Mexico): 4.2 vs 4.1 → 4.2
    assert best["draw"] == 3.5
    assert best["away"] == 4.2


def test_best_book_for_identifica_la_casa():
    matches = load_sample(SAMPLE)
    # La mejor cuota de Argentina (2.00) la da Pinnacle.
    assert best_book_for(matches[0], "Argentina") == "Pinnacle"


def test_parse_events_maneja_partido_sin_cuotas():
    raw = [
        {
            "id": "x",
            "home_team": "A",
            "away_team": "B",
            "commence_time": "2026-06-20T18:00:00Z",
            "bookmakers": [],
        }
    ]
    matches = parse_events(raw)
    assert len(matches) == 1
    assert matches[0].books_h2h == {}
    assert best_1x2(matches[0]) == {}
