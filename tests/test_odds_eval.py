"""Tests del lector de cuotas + evaluador (cuotas buenas + combinadas)."""

from __future__ import annotations

from mundial_bot.collectors.odds_af import parse_odds
from mundial_bot.evaluator import GoodBet, build_parlays, evaluate_match
from mundial_bot.report import MarketPick, MatchReport

ODDS_RAW = {
    "response": [
        {
            "fixture": {"id": 1},
            "bookmakers": [
                {"name": "Bet365", "bets": [
                    {"name": "Match Winner", "values": [
                        {"value": "Home", "odd": "3.20"},
                        {"value": "Draw", "odd": "2.90"},
                        {"value": "Away", "odd": "2.40"},
                    ]},
                ]},
                {"name": "Pinnacle", "bets": [
                    {"name": "Match Winner", "values": [
                        {"value": "Away", "odd": "2.65"},  # mejor que Bet365
                    ]},
                    {"name": "Goals Over/Under", "values": [
                        {"value": "Over 2.5", "odd": "1.95"},
                        {"value": "Under 2.5", "odd": "1.85"},
                    ]},
                ]},
            ],
        }
    ]
}


def test_parse_odds_toma_la_mejor_cuota():
    odds = parse_odds(ODDS_RAW)
    assert "Match Winner" in odds
    # Away: Bet365 2.40 vs Pinnacle 2.65 → mejor 2.65.
    assert odds["Match Winner"].best["Away"] == (2.65, "Pinnacle")


def _report(match: str, away_prob: float) -> MatchReport:
    return MatchReport(
        match=match, home_prob=0.30, draw_prob=0.25, away_prob=away_prob,
        winner=MarketPick("Gana visitante", away_prob, side="away"),
        goals=MarketPick("Over 2.5 goles", 0.55, expected=2.8, side="over", line=2.5),
        btts=None, corners=None, cards=None,
    )


def test_evaluate_match_encuentra_cuotas_buenas():
    odds = parse_odds(ODDS_RAW)
    bets = evaluate_match(_report("A vs B", away_prob=0.45), odds, min_ev=0.0)

    # Visitante: 0.45 * 2.65 - 1 = +0.19 → buena. Over 2.5: 0.55*1.95-1 = +0.07 → buena.
    markets = {b.market for b in bets}
    assert "Ganador" in markets
    assert "Goles" in markets
    winner = next(b for b in bets if b.market == "Ganador")
    assert winner.ev > 0.15
    assert winner.book == "Pinnacle"


def test_evaluate_match_descarta_cuotas_malas():
    odds = parse_odds(ODDS_RAW)
    # Visitante con prob baja (0.30): 0.30*2.65-1 = -0.20 → no es buena.
    bets = evaluate_match(_report("A vs B", away_prob=0.30), odds, min_ev=0.0)
    assert all(b.market != "Ganador" for b in bets)


def test_build_parlays_combina_partidos_distintos():
    b1 = GoodBet("A vs B", "Ganador", "Gana B", 0.55, 2.0, "x")
    b2 = GoodBet("C vs D", "Ganador", "Gana C", 0.55, 2.0, "y")
    parlays = build_parlays([b1, b2], sizes=(2,))
    assert len(parlays) == 1
    assert parlays[0].combined_odds == 4.0
    assert parlays[0].ev > 0  # 0.3025*4 - 1 = +0.21


def test_build_parlays_no_combina_mismo_partido():
    b1 = GoodBet("A vs B", "Ganador", "Gana B", 0.6, 2.0, "x")
    b2 = GoodBet("A vs B", "Goles", "Over 2.5", 0.6, 2.0, "y")
    assert build_parlays([b1, b2], sizes=(2,)) == []
