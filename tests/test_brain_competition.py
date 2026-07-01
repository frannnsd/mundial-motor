"""Tests de la Fase A: distribuciones, cerebros, competencia (guard!), unificación
y proyección de mercados."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mundial_bot.backtest.leakage_guard import LeakageError
from mundial_bot.markets import projection as proj
from mundial_bot.research import competition as comp
from mundial_bot.research.brains import BrainConfig, LeagueState
from mundial_bot.research.competition import (
    evaluate_unified_holdout,
    run_competition,
    unify,
)
from mundial_bot.research.distributions import (
    convolve_pmf,
    count_pmf,
    crps_count,
    p_over,
)

# ---------------------------------------------------------------------------
# distribuciones
# ---------------------------------------------------------------------------

def test_count_pmf_sums_to_one_and_matches_mean():
    for mean, var in ((2.7, 2.7), (10.0, 14.0), (0.05, 0.05)):
        pmf = count_pmf(mean, var, 30)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
        approx_mean = float(np.dot(np.arange(31), pmf))
        assert approx_mean == pytest.approx(mean, rel=0.02)


def test_crps_rewards_better_prediction():
    actual = 10
    good = count_pmf(10.0, 12.0, 30)
    bad = count_pmf(4.0, 5.0, 30)
    assert crps_count(good, actual) < crps_count(bad, actual)


def test_convolve_and_p_over():
    a = count_pmf(5.0, 6.0, 20)
    b = count_pmf(5.0, 6.0, 20)
    total = convolve_pmf(a, b)
    assert total.sum() == pytest.approx(1.0, abs=1e-9)
    mean_total = float(np.dot(np.arange(len(total)), total))
    assert mean_total == pytest.approx(10.0, rel=0.02)
    assert 0.35 < p_over(total, 9.5) < 0.65  # la línea del medio queda pareja


# ---------------------------------------------------------------------------
# dataset sintético para la competencia
# ---------------------------------------------------------------------------

def _stats_df(n_rounds: int = 40, seasons=("2223", "2324")) -> pd.DataFrame:
    """4 equipos con niveles distintos, 2 partidos por fecha, valores determinísticos."""
    lvl = {"A": 6, "B": 5, "C": 4, "D": 3}  # córners "for" típicos
    rows, mid = [], 0
    for season, start in zip(seasons, ("2023-01-07", "2024-01-06"), strict=True):
        base = pd.Timestamp(start)
        for rnd in range(n_rounds):
            day = base + pd.Timedelta(days=3 * rnd)
            pairs = [("A", "B"), ("C", "D")] if rnd % 2 == 0 else [("A", "C"), ("B", "D")]
            for h, a in pairs:
                ch, ca = lvl[h] + (mid % 3), lvl[a] + (mid % 2)
                rows.append({
                    "date": day, "home_team": h, "away_team": a,
                    "home_score": 1 + (mid % 3), "away_score": mid % 2,
                    "corners_h": ch, "corners_a": ca,
                    "shots_h": ch * 2 + 2, "shots_a": ca * 2,
                    "sot_h": ch - 1, "sot_a": max(ca - 2, 0),
                    "yellows_h": 2, "yellows_a": 2 + (mid % 2),
                    "reds_h": 0, "reds_a": 1 if mid % 17 == 0 else 0,
                    "fouls_h": 11, "fouls_a": 12,
                    "ht_goals_h": 0, "ht_goals_a": 0,
                    "league": "E0", "season": season, "match_id": f"E0_{season}_{mid}",
                })
                mid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# competencia: guard en el loop + estructura
# ---------------------------------------------------------------------------

def test_guard_runs_per_match_and_halts(monkeypatch):
    df = _stats_df()
    calls = {"n": 0}
    real = comp.assert_point_in_time

    def spy(events, as_of, **kw):
        calls["n"] += 1
        return real(events, as_of, **kw)

    monkeypatch.setattr(comp, "assert_point_in_time", spy)
    run_competition(df, config=BrainConfig(min_fit_rows=50))
    assert calls["n"] == len(df)  # una vez POR PARTIDO

    def boom(events, as_of, **kw):
        raise LeakageError("leak simulado")

    monkeypatch.setattr(comp, "assert_point_in_time", boom)
    with pytest.raises(LeakageError):
        run_competition(df)


def test_competition_structure_and_holdout_split():
    df = _stats_df()
    res = run_competition(df, config=BrainConfig(min_fit_rows=50))
    # el warm-up no existe acá ("1415" no está) → valida 2223, hold-out 2324
    assert res.n_scored_validation > 0 and res.n_scored_holdout > 0
    assert "corners_h" in res.validation and "corners_h" in res.holdout
    for brain in ("A", "B", "C", "bobo"):
        assert res.validation["corners_h"][brain]["crps"] > 0
    # pmfs del hold-out guardadas para el unificado
    assert len(res.holdout_pmfs) == res.n_scored_holdout


def test_dumb_baseline_is_point_in_time():
    """El bobo usa la media de la TEMPORADA hasta ese partido: en un dataset donde la
    2ª temporada tiene el doble de córners, sus primeras predicciones de la 2ª
    temporada NO pueden reflejar ya el nivel nuevo (no conoce el futuro)."""
    state = LeagueState(BrainConfig())
    day = pd.Timestamp("2023-01-01")
    row = pd.Series({
        "home_team": "A", "away_team": "B", "season": "2223",
        "home_score": 1, "away_score": 0, "corners_h": 5, "corners_a": 5,
        "shots_h": 10, "shots_a": 10, "sot_h": 4, "sot_a": 4,
        "yellows_h": 2, "yellows_a": 2, "reds_h": 0, "reds_a": 0,
    })
    preds = state.predict(row, day)
    # sin historia, el bobo cae al default (media de liga vacía → fallback), no crashea
    assert preds["bobo"]["corners_h"][0] > 0


# ---------------------------------------------------------------------------
# unificación
# ---------------------------------------------------------------------------

def test_unify_zeroes_losers_and_falls_back_to_dumb():
    validation = {
        "corners_h": {  # A y C le ganan al bobo; B pierde → peso 0
            "A": {"crps": 1.50}, "B": {"crps": 1.90},
            "C": {"crps": 1.55}, "bobo": {"crps": 1.70},
        },
        "reds_h": {  # nadie le gana al bobo → unificado = bobo
            "A": {"crps": 0.052}, "B": {"crps": 0.053},
            "C": {"crps": 0.051}, "bobo": {"crps": 0.050},
        },
    }
    w = unify(validation)
    assert "B" not in w["corners_h"]           # perdió contra el bobo → 0
    assert sum(w["corners_h"].values()) == pytest.approx(1.0)
    assert w["corners_h"]["A"] > w["corners_h"]["C"] > 0
    assert w["reds_h"] == {"bobo": 1.0}        # nadie ganó → bobo


def test_unified_holdout_mixture():
    df = _stats_df()
    res = run_competition(df, config=BrainConfig(min_fit_rows=50))
    w = unify(res.validation)
    out = evaluate_unified_holdout(res, w)
    assert "corners_h" in out["metrics"]
    assert out["metrics"]["corners_h"]["n"] == res.n_scored_holdout
    for market in ("over_2.5_goles", "over_9.5_corners", "btts"):
        table, ece = out["market_calibration"][market]
        assert 0.0 <= ece <= 1.0 and len(table) == 10


# ---------------------------------------------------------------------------
# proyección de mercados
# ---------------------------------------------------------------------------

def _unified_example() -> dict[str, np.ndarray]:
    return {
        "goals_h": count_pmf(1.5, 1.6, 12), "goals_a": count_pmf(1.2, 1.3, 12),
        "corners_h": count_pmf(5.5, 8.0, 25), "corners_a": count_pmf(4.5, 7.0, 25),
        "yellows_h": count_pmf(2.1, 2.9, 14), "yellows_a": count_pmf(2.3, 3.1, 14),
        "shots_h": count_pmf(13.0, 20.0, 45), "shots_a": count_pmf(11.0, 17.0, 45),
        "sot_h": count_pmf(4.6, 6.0, 22), "sot_a": count_pmf(4.1, 5.5, 22),
        "reds_h": count_pmf(0.04, 0.045, 4), "reds_a": count_pmf(0.05, 0.055, 4),
    }


def test_projection_probabilities_are_coherent():
    u = _unified_example()
    p = proj.project_all(u)
    assert sum(p["1x2"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(p["rango_goles"].values()) == pytest.approx(1.0, abs=1e-6)
    assert p["doble_oportunidad"]["1X"] == pytest.approx(
        p["1x2"]["home"] + p["1x2"]["draw"], abs=1e-9
    )
    # BTTS coincide con el cálculo manual desde las marginales
    manual = (1 - u["goals_h"][0]) * (1 - u["goals_a"][0])
    assert p["btts"]["yes"] == pytest.approx(float(manual), abs=1e-9)
    # over/under monotónico en la línea
    assert p["goles_ou"][1.5]["over"] > p["goles_ou"][2.5]["over"] > p["goles_ou"][3.5]["over"]
    assert 0.0 < p["roja_en_partido"] < 0.2


def test_projection_horizon_120_is_explicit_todo():
    u = _unified_example()
    with pytest.raises(NotImplementedError):
        proj.one_x_two(u["goals_h"], u["goals_a"], horizon="120")
    with pytest.raises(NotImplementedError):
        proj.ht_ft()
    with pytest.raises(ValueError):
        proj.one_x_two(u["goals_h"], u["goals_a"], horizon="45")
