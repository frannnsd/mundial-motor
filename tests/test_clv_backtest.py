"""Tests del backtest de CLV (Fase 2).

Cubre: estructura del resultado, la matemática del CLV, y las DOS innegociables —
(b) el guard anti-leakage corre DENTRO del loop por partido y CORTA el backtest si falla.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.backtest import clv_backtest as mod
from mundial_bot.backtest.clv_backtest import _score_match, run_clv_backtest
from mundial_bot.backtest.leakage_guard import LeakageError


def _synth_df() -> pd.DataFrame:
    """Mini dataset con el esquema de football_data: 1 liga, 2 temporadas (dev + hold-out)."""
    rows = []
    seasons = [("2223", "2023-01-07"), ("2324", "2024-01-06")]
    mid = 0
    for season, start in seasons:
        base = pd.Timestamp(start)
        # 6 partidos por temporada, en 3 fechas (2 por fecha → prueba el batching same-day)
        pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
        for gi in range(3):
            day = base + pd.Timedelta(days=7 * gi)
            for pi in range(2):
                h, a = pairs[gi * 2 + pi]
                hs, as_ = (2, 0) if (mid % 2 == 0) else (1, 1)
                rows.append({
                    "date": day, "home_team": h, "away_team": a,
                    "home_score": hs, "away_score": as_,
                    "tournament": "England Premier League", "neutral": False,
                    "league": "E0", "season": season,
                    "psc_h": 1.90, "psc_d": 3.50, "psc_a": 4.20,      # cierre
                    "ps_h": 2.05, "ps_d": 3.40, "ps_a": 4.00,         # apertura
                    "match_id": f"E0_{season}_{mid}",
                })
                mid += 1
    return pd.DataFrame(rows)


def test_result_structure_and_metrics():
    res = run_clv_backtest(_synth_df())
    assert res["n"] > 0
    assert "holdout" in res and "dev" in res
    ho = res["holdout"]
    assert ho["n"] >= 1
    assert "brier_model" in ho and "brier_close" in ho
    assert "logloss_model" in ho and "logloss_close" in ho
    if ho.get("n_bets"):
        assert -1.0 <= ho["clv_mean"] <= 5.0
        assert 0.0 <= ho["pct_beat_close"] <= 1.0


def test_score_match_clv_math():
    row = pd.Series({
        "home_score": 2, "away_score": 0, "season": "2324", "league": "E0",
        "psc_h": 1.80, "psc_d": 3.60, "psc_a": 4.50,   # cierre
        "ps_h": 2.00, "ps_d": 3.50, "ps_a": 4.20,      # apertura: mejor precio en home
    })
    rec = _score_match(row, {"home": 0.70, "draw": 0.20, "away": 0.10}, min_edge=0.0, method="shin")
    assert rec["actual"] == "home"
    assert rec["bet"] == "home"                       # el modelo ve más valor en home
    assert rec["clv"] == pytest.approx(2.00 / 1.80 - 1.0)   # apertura/cierre − 1
    assert rec["won"] is True
    assert rec["roi"] == pytest.approx(2.00 - 1.0)


def test_score_match_no_open_odds_no_bet():
    row = pd.Series({
        "home_score": 1, "away_score": 1, "season": "2223", "league": "E0",
        "psc_h": 2.10, "psc_d": 3.30, "psc_a": 3.60,
        "ps_h": float("nan"), "ps_d": float("nan"), "ps_a": float("nan"),  # sin apertura
    })
    rec = _score_match(row, {"home": 0.5, "draw": 0.3, "away": 0.2}, min_edge=0.0, method="shin")
    assert rec is not None
    assert rec["bet"] is None and rec["clv"] is None   # calibración sí, CLV no


# --- INNEGOCIABLE (b): el guard corre por partido y corta si falla ---

def test_guard_runs_in_loop_per_match(monkeypatch):
    calls = {"n": 0}
    real = mod.assert_point_in_time

    def spy(events, as_of, **kw):
        calls["n"] += 1
        return real(events, as_of, **kw)

    monkeypatch.setattr(mod, "assert_point_in_time", spy)
    df = _synth_df()
    run_clv_backtest(df)
    assert calls["n"] == len(df)   # exactamente una vez por partido


def test_guard_halts_backtest_on_leak(monkeypatch):
    state = {"n": 0}

    def boom(events, as_of, **kw):
        state["n"] += 1
        if state["n"] == 3:
            raise LeakageError("leak simulado dentro del loop")

    monkeypatch.setattr(mod, "assert_point_in_time", boom)
    with pytest.raises(LeakageError):
        run_clv_backtest(_synth_df())
    assert state["n"] == 3          # se cortó apenas falló, no siguió
