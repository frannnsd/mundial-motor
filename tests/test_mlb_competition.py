"""Tests M1+M2 MLB: estado point-in-time, guard en el loop, proyección coherente."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.backtest.leakage_guard import LeakageError
from mundial_bot.markets import mlb_projection as proj
from mundial_bot.research import mlb as mlbmod
from mundial_bot.research.distributions import count_pmf
from mundial_bot.research.mlb import (
    MlbConfig,
    MlbState,
    evaluate_unified_mlb,
    run_mlb_competition,
    unify,
)


def _games(n_days: int = 30, seasons=(("2024", "2024-04-01"), ("2025", "2025-04-01"))):
    rows, pk = [], 1000
    teams = [("Yankees", "Red Sox", "Stadium A", 11, 22),
             ("Dodgers", "Padres", "Stadium B", 33, 44)]
    for season, start in seasons:
        base = pd.Timestamp(start)
        for d in range(n_days):
            day = base + pd.Timedelta(days=d)
            for home, away, venue, sh, sa in teams:
                rows.append({
                    "date": day, "game_pk": pk, "home_team": home, "away_team": away,
                    "venue": venue, "runs_h": 4 + (pk % 3), "runs_a": 3 + (pk % 2),
                    "hits_h": 8 + (pk % 4), "hits_a": 7 + (pk % 3),
                    "runs_f5_h": 2 + (pk % 2), "runs_f5_a": 2,
                    "starter_h_id": sh, "starter_h": f"P{sh}",
                    "starter_a_id": sa, "starter_a": f"P{sa}",
                    "season": season, "league": "MLB", "match_id": str(pk),
                })
                pk += 1
    return pd.DataFrame(rows)


def test_predict_does_not_mutate_state():
    st = MlbState(MlbConfig())
    df = _games(5)
    row, day = df.iloc[0], df["date"].iloc[0]
    before = (len(st.teams), len(st.starters), st._mom("runs", "h").n)
    st.predict(row, day)
    after = (len(st.teams), len(st.starters), st._mom("runs", "h").n)
    # predict crea structs de equipos vacíos pero NO agrega observaciones ni pitchers
    assert before[2] == after[2] == 0.0
    assert after[1] == 0


def test_starter_matters_for_brain_b():
    """Un abridor castigado debe subir la media del rival en B (no en A)."""
    cfg = MlbConfig()
    st = MlbState(cfg)
    df = _games(40, seasons=(("2024", "2024-04-01"),))
    day0 = df["date"].iloc[0]
    # calentar el estado con todos los partidos
    for _, row in df.iterrows():
        st.reveal(row, row["date"])
    st.end_day(df["date"].iloc[-1])
    future = df["date"].iloc[-1] + pd.Timedelta(days=1)
    # castigar a un pitcher nuevo: le anotaron 9 por partido
    bad_id = 999
    for k in range(8):
        st.starters.setdefault(bad_id, {}).setdefault("runs", mlbmod._Decayed()).update(
            9.0, day0 + pd.Timedelta(days=k), cfg.halflife_days
        )
    base_row = pd.Series({
        "home_team": "Yankees", "away_team": "Red Sox", "season": "2024",
        "venue": "Stadium A", "starter_h_id": 11, "starter_a_id": 22,
    })
    bad_row = base_row.copy()
    bad_row["starter_a_id"] = bad_id  # el visitante pone al pitcher malo
    p_base = st.predict(base_row, future)
    p_bad = st.predict(bad_row, future)
    st._pending.clear()
    assert p_bad["B"]["runs_h"][0] > p_base["B"]["runs_h"][0]      # B lo castiga
    assert p_bad["A"]["runs_h"][0] == pytest.approx(p_base["A"]["runs_h"][0])  # A no


def test_guard_runs_per_game_and_halts(monkeypatch):
    df = _games(10)
    calls = {"n": 0}
    real = mlbmod.assert_point_in_time

    def spy(events, as_of, **kw):
        calls["n"] += 1
        return real(events, as_of, **kw)

    monkeypatch.setattr(mlbmod, "assert_point_in_time", spy)
    run_mlb_competition(df)
    assert calls["n"] == len(df)

    def boom(events, as_of, **kw):
        raise LeakageError("leak")

    monkeypatch.setattr(mlbmod, "assert_point_in_time", boom)
    with pytest.raises(LeakageError):
        run_mlb_competition(df)


def test_competition_splits_and_unified():
    df = _games(40)  # 2024 validación, 2025 hold-out (sin warm-up en el synthetic)
    res = run_mlb_competition(df)
    assert res["n_validation"] > 0 and res["n_holdout"] > 0
    w = unify(res["validation"])
    assert all(abs(sum(v.values()) - 1) < 1e-9 for v in w.values())
    uni = evaluate_unified_mlb(res, w)
    assert "runs_h" in uni["metrics"]
    for m, (table, ece) in uni["market_calibration"].items():
        assert 0 <= ece <= 1 and len(table) == 10, m


def test_projection_mlb_coherent():
    pmfs = {
        "runs_h": count_pmf(4.6, 6.5, 22), "runs_a": count_pmf(4.2, 6.0, 22),
        "hits_h": count_pmf(8.6, 10.0, 32), "hits_a": count_pmf(8.1, 9.5, 32),
        "runs_f5_h": count_pmf(2.6, 3.4, 16), "runs_f5_a": count_pmf(2.4, 3.2, 16),
    }
    p = proj.project_all_mlb(pmfs)
    ml = p["moneyline"]
    assert ml["home"] + ml["away"] == pytest.approx(1.0, abs=1e-9)
    assert ml["home"] > 0.5  # el local con más carreras esperadas es favorito
    rl = p["run_line"]
    assert rl["home_-1.5"] + rl["away_+1.5"] == pytest.approx(1.0, abs=1e-9)
    assert rl["home_-1.5"] < ml["home"]  # cubrir -1.5 es más difícil que ganar
    t = p["totales"]
    assert t["7.5"]["over"] > t["8.5"]["over"] > t["9.5"]["over"]
    f5 = p["f5"]["ml_3way"]
    assert f5["home"] + f5["tie"] + f5["away"] == pytest.approx(1.0, abs=1e-9)
    assert p["f5"]["totales"]["4.5"]["over"] < t["8.5"]["over"] + 0.5  # sanity blanda
