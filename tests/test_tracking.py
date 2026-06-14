"""Tests del sistema de autoevaluación (grading + balance)."""

from __future__ import annotations

from mundial_bot.report import MarketPick, MatchReport
from mundial_bot.tracking import PredictionStore, format_balance, grade_outcome


def test_grade_ganador():
    win = {"home_score": 2, "away_score": 0}
    assert grade_outcome("winner", "home", None, win)[0] == "correct"
    assert grade_outcome("winner", "away", None, win)[0] == "wrong"
    draw = {"home_score": 1, "away_score": 1}
    assert grade_outcome("winner", "draw", None, draw)[0] == "correct"


def test_grade_goles_over_under():
    res = {"home_score": 2, "away_score": 1}  # total 3
    assert grade_outcome("goals", "over", 2.5, res)[0] == "correct"
    assert grade_outcome("goals", "under", 2.5, res)[0] == "wrong"


def test_grade_corners_y_tarjetas():
    res = {"home_score": 0, "away_score": 0, "corners": 11, "cards": 4}
    assert grade_outcome("corners", "over", 9.5, res)[0] == "correct"
    assert grade_outcome("corners", "under", 9.5, res)[0] == "wrong"
    assert grade_outcome("cards", "under", 4.5, res)[0] == "correct"


def test_grade_btts():
    assert grade_outcome("btts", "yes", None, {"home_score": 1, "away_score": 2})[0] == "correct"
    assert grade_outcome("btts", "no", None, {"home_score": 1, "away_score": 0})[0] == "correct"


def _report() -> MatchReport:
    return MatchReport(
        match="A vs B", home_prob=0.6, draw_prob=0.2, away_prob=0.2,
        winner=MarketPick("Gana A", 0.6, side="home"),
        goals=MarketPick("Over 2.5 goles", 0.55, expected=2.8, side="over", line=2.5),
        corners=MarketPick("Over 9.5 córners", 0.6, expected=10.0, side="over", line=9.5),
        cards=None, btts=None,
    )


def test_log_es_idempotente_y_balance_calcula():
    store = PredictionStore(":memory:")
    try:
        n = store.log_report(123, _report(), pred_date="2026-06-14", created_at="2026-06-14")
        assert n == 3  # winner + goals + corners
        # Re-loguear el mismo día no duplica.
        again = store.log_report(123, _report(), pred_date="2026-06-14", created_at="2026-06-14")
        assert again == 0

        for pred in store.pending_for_fixture(123):
            ok = pred["market"] == "winner"
            store.settle(pred["id"], status="correct" if ok else "wrong", actual=1.0)

        bal = store.balance()
        assert bal.n == 3
        assert bal.correct == 1
        assert bal.by_market["winner"] == (1, 1)
        assert "Aciertos" in format_balance(bal)
    finally:
        store.close()
