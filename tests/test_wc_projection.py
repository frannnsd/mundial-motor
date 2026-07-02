"""Tests del horizonte 120' (prórroga condicional) y mercados de eliminatoria."""

from __future__ import annotations

import numpy as np
import pytest

from mundial_bot.markets import projection as proj
from mundial_bot.research.distributions import count_pmf


def _pmfs(mh: float = 1.5, ma: float = 1.2) -> dict[str, np.ndarray]:
    return {
        "goals_h": count_pmf(mh, mh * 1.05, 12), "goals_a": count_pmf(ma, ma * 1.05, 12),
        "corners_h": count_pmf(5.5, 8.0, 25), "corners_a": count_pmf(4.5, 7.0, 25),
        "yellows_h": count_pmf(2.1, 2.9, 14), "yellows_a": count_pmf(2.3, 3.1, 14),
        "shots_h": count_pmf(13.0, 20.0, 45), "shots_a": count_pmf(11.0, 17.0, 45),
        "sot_h": count_pmf(4.6, 6.0, 22), "sot_a": count_pmf(4.1, 5.5, 22),
        "reds_h": count_pmf(0.04, 0.045, 4), "reds_a": count_pmf(0.05, 0.055, 4),
    }


def test_extra_time_prob_equals_draw_probability():
    u = _pmfs()
    p_et = proj.extra_time_prob(u["goals_h"], u["goals_a"])
    assert p_et == pytest.approx(proj.one_x_two(u["goals_h"], u["goals_a"])["draw"], abs=1e-9)
    assert 0.1 < p_et < 0.5


def test_pmf_te_mean_and_normalization():
    u = _pmfs()
    p_et = proj.extra_time_prob(u["goals_h"], u["goals_a"])
    te = proj.pmf_te(u["corners_h"], p_et)
    assert te.sum() == pytest.approx(1.0, abs=1e-9)
    mean90 = float(np.dot(np.arange(len(u["corners_h"])), u["corners_h"]))
    mean_te = float(np.dot(np.arange(len(te)), te))
    expected = mean90 * (1.0 + p_et * (30.0 / 90.0) * proj.ET_FATIGUE)
    assert mean_te == pytest.approx(expected, rel=0.01)  # ~exacto salvo truncamiento
    # y coincide con el helper que usa la capa de props
    assert proj.team_total_at_horizon(mean90, p_et) == pytest.approx(expected, rel=1e-9)


def test_to_qualify_partitions_and_symmetry():
    u = _pmfs()
    q = proj.to_qualify(u["goals_h"], u["goals_a"])
    assert q["home"] + q["away"] == pytest.approx(1.0, abs=1e-6)
    assert q["home"] > q["away"]  # el local es mejor en este ejemplo
    sym = _pmfs(1.3, 1.3)
    qs = proj.to_qualify(sym["goals_h"], sym["goals_a"])
    assert qs["home"] == pytest.approx(0.5, abs=1e-6)  # equipos iguales + penales 50/50


def test_method_of_victory_sums_to_one():
    u = _pmfs()
    m = proj.method_of_victory(u["goals_h"], u["goals_a"])
    assert sum(m.values()) == pytest.approx(1.0, abs=1e-6)
    assert m["home_90"] > m["home_et"] > 0
    assert m["home_pens"] > 0 and m["away_pens"] > 0


def test_knockout_markets_te_over_exceeds_90():
    u = _pmfs()
    ko = proj.knockout_markets(u)
    assert 0.0 < ko["p_prorroga"] < 1.0
    over_90 = proj.total_over_under(u["corners_h"], u["corners_a"], 9.5)["over"]
    over_te = ko["te"]["corners"][9.5]["over"]
    assert over_te > over_90  # más minutos posibles → más córners esperados
    assert set(ko["se_clasifica"]) == {"home", "away"}


def test_props_coherence_at_te_horizon():
    """La coherencia del reparto por jugador se mantiene con totales TE."""
    import pandas as pd

    from mundial_bot.players.props import match_props

    u = _pmfs()
    p_et = proj.extra_time_prob(u["goals_h"], u["goals_a"])
    mean90_shots = float(np.dot(np.arange(len(u["shots_h"])), u["shots_h"]))
    total_te = proj.team_total_at_horizon(mean90_shots, p_et)

    shares = pd.DataFrame({
        "player_id": [1, 2, 3], "player_name": ["A", "B", "C"], "position": ["F", "M", "D"],
        "min_avg": [85.0, 80.0, 90.0],
        "min_avg_starter": [85.0, 80.0, 90.0], "min_avg_sub": [0.0, 0.0, 0.0],
        "rate_shots": [3.0, 1.5, 0.5],
    })
    team_totals = {"shots": (total_te, total_te * 1.4)}
    out = match_props(team_totals, shares, horizon="120")
    assert out["mu_shots"].sum() == pytest.approx(total_te, abs=1e-6)


def test_regulation_markets_still_reject_120():
    u = _pmfs()
    with pytest.raises(NotImplementedError):
        proj.one_x_two(u["goals_h"], u["goals_a"], horizon="120")
