"""Tests del lector de cuotas + escáner probabilístico (sin value)."""

from __future__ import annotations

from mundial_bot.collectors.odds_af import parse_odds
from mundial_bot.evaluator import Play, best_plays, build_combos, scan_match
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


def _report(match: str, away_prob: float) -> MatchReport:
    return MatchReport(
        match=match, home_prob=0.30, draw_prob=0.25, away_prob=away_prob,
        winner=MarketPick("Gana visitante", away_prob, side="away"),
        goals=MarketPick("Over 2.5 goles", 0.55, expected=2.8, side="over", line=2.5),
        btts=None, corners=None, cards=None,
    )


def test_parse_odds_toma_la_mejor_cuota():
    odds = parse_odds(ODDS_RAW)
    # Away: Bet365 2.40 vs Pinnacle 2.65 → mejor 2.65.
    assert odds["Match Winner"].best["Away"] == (2.65, "Pinnacle")


def test_parse_odds_filtra_a_las_casas_de_franco():
    # Filtrando a Bet365, la mejor de Away ya no es Pinnacle (2.65) sino Bet365 (2.40).
    odds = parse_odds(ODDS_RAW, books={"bet365"})
    assert odds["Match Winner"].best["Away"] == (2.40, "Bet365")
    # Pinnacle quedó afuera → su mercado Goals Over/Under no aparece.
    assert "Goals Over/Under" not in odds


def test_scan_match_adjunta_la_cuota_y_no_descarta():
    odds = parse_odds(ODDS_RAW)
    plays = scan_match(_report("A vs B", away_prob=0.45), odds)
    by_market = {p.market: p for p in plays}

    winner = by_market["Ganador"]
    assert winner.pick == "Gana visitante"
    assert winner.odd == 2.65 and winner.book == "Pinnacle"
    assert winner.prob == 0.45

    goles = by_market["Goles"]
    assert goles.odd == 1.95
    assert goles.implied == 1.0 / 1.95


def test_scan_match_muestra_aunque_la_cuota_pague_poco():
    # Sin "value": la jugada poco probable IGUAL se muestra (Franco decide).
    odds = parse_odds(ODDS_RAW)
    plays = scan_match(_report("A vs B", away_prob=0.30), odds)
    winner = next(p for p in plays if p.market == "Ganador")
    assert winner.odd == 2.65          # se muestra igual
    assert winner.prob == 0.30


def test_scan_match_sin_cuota_listada_usa_cuota_del_modelo():
    plays = scan_match(_report("A vs B", away_prob=0.50), odds={})
    winner = next(p for p in plays if p.market == "Ganador")
    assert winner.odd is None
    assert winner.model_odds == 2.0    # 1 / 0.50


def test_build_combos_mas_probables_y_de_mayor_pago():
    p1 = Play("A vs B", "Ganador", "Gana B", 0.60, 2.0, "x")
    p2 = Play("C vs D", "Ganador", "Gana C", 0.55, 3.0, "y")
    likely, payout = build_combos([p1, p2], sizes=(2,))
    assert len(likely) == 1
    assert likely[0].combined_odds == 6.0
    assert likely[0].combined_prob == 0.60 * 0.55
    assert payout[0].combined_odds == 6.0


def test_build_combos_no_combina_mismo_partido():
    p1 = Play("A vs B", "Ganador", "Gana B", 0.6, 2.0, "x")
    p2 = Play("A vs B", "Goles", "Over 2.5", 0.6, 2.0, "y")
    likely, payout = build_combos([p1, p2], sizes=(2,))
    assert likely == [] and payout == []


def test_build_combos_ignora_patas_sin_cuota():
    p1 = Play("A vs B", "Ganador", "Gana B", 0.6, 2.0, "x")
    p2 = Play("C vs D", "Córners", "Over 8.5", 0.7, None, "")  # sin cuota
    likely, _ = build_combos([p1, p2], sizes=(2,))
    assert likely == []


def test_build_combos_dedup_no_repite_patas():
    p1 = Play("A vs B", "Ganador", "Gana B", 0.6, 2.0, "x")
    p2 = Play("C vs D", "Ganador", "Gana C", 0.6, 2.0, "y")
    likely, _ = build_combos([p1, p2, p1], sizes=(2,))   # p1 repetida
    assert len(likely) == 1


def test_best_plays_descarta_cuota_rota():
    # modelo 99% pero la casa paga 13 (implícita ~8%): gap imposible → dato roto, fuera.
    roto = Play("A vs B", "Hándicap asiático", "B +2", 0.99, 13.0, "x")
    firme = Play("C vs D", "Ganador", "Gana C", 0.70, 1.60, "y")
    firmes, mejor, batacazos = best_plays([roto, firme])
    assert roto not in (firmes + mejor + batacazos)
    assert firme in firmes


def test_best_plays_batacazo_es_poco_probable_y_paga_fuerte():
    bat = Play("A vs B", "Ganador", "Empate", 0.28, 6.0, "x")
    _, _, batacazos = best_plays([bat])
    assert bat in batacazos
