"""Tests del pipeline diario (sin red: todo mockeado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mundial_bot.forward_test import log as ft
from mundial_bot.research.distributions import count_pmf
from mundial_bot.wc import daily


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "ft.sqlite"
    monkeypatch.setattr(ft, "FORWARD_TEST_DB", db)
    return db


def _fx(fid=999, home="United States", away="Bosnia and Herzegovina",
        status="NS", rnd="Round of 32"):
    return {
        "fixture": {"id": fid, "date": "2026-07-02T20:00:00+00:00",
                    "status": {"short": status}},
        "league": {"round": rnd},
        "teams": {"home": {"name": home, "winner": None},
                  "away": {"name": away, "winner": None}},
    }


class _FakeEngine:
    def predict_match(self, home, away, *, when, neutral=None):
        means = {"goals_h": 1.6, "goals_a": 1.1, "corners_h": 5.2, "corners_a": 4.4,
                 "yellows_h": 2.0, "yellows_a": 2.2, "shots_h": 13.0, "shots_a": 10.5,
                 "sot_h": 4.5, "sot_a": 3.8, "reds_h": 0.04, "reds_a": 0.05}
        grids = {"goals": 12, "corners": 25, "yellows": 14, "shots": 45,
                 "sot": 22, "reds": 4}
        pmfs = {}
        for fam, g in grids.items():
            for side in ("h", "a"):
                m = means[f"{fam}_{side}"]
                pmfs[f"{fam}_{side}"] = count_pmf(m, m * 1.3, g)
        return {"pmfs": pmfs, "means": means, "weights": {}, "neutral": True}


def _fake_props(_table, team, *_a, **_k):
    base = 10 if team.startswith("United") else 20  # ids distintos por equipo
    return pd.DataFrame({
        "player_id": [base, base + 1], "player_name": ["Balogun", "Pepi"],
        "position": ["F", "F"], "exp_minutes": [80.0, 65.0],
        "mu_shots": [1.7, 1.3], "mu_sot": [0.7, 0.5],
        "p_scores": [0.32, 0.22], "p_card": [0.15, 0.12],
    })


def test_pre_day_writes_report_and_logs(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "_day_fixtures", lambda *a, **k: [_fx()])
    monkeypatch.setattr(daily, "_build_engine", lambda: _FakeEngine())
    monkeypatch.setattr(daily, "_player_table", lambda: pd.DataFrame())
    monkeypatch.setattr(daily, "_props_for", _fake_props)
    monkeypatch.setattr(daily, "REPORTS_DAILY", tmp_path / "daily")
    monkeypatch.setattr(daily.get_settings(), "api_football_key", "x", raising=False)

    daily.cmd_pre_day("2026-07-02")

    md = (tmp_path / "daily" / "2026-07-02.md").read_text(encoding="utf-8")
    assert "United States vs Bosnia" in md
    assert "1X2:" in md and "se clasifica" in md and "Props" in md
    assert "Totales TE" in md  # es eliminatoria → mercados TE presentes
    s = ft.summary()
    assert s["total"] >= 15  # toda predicción emitida quedó logueada


def test_pre_day_idempotent_no_duplicates(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "_day_fixtures", lambda *a, **k: [_fx()])
    monkeypatch.setattr(daily, "_build_engine", lambda: _FakeEngine())
    monkeypatch.setattr(daily, "_player_table", lambda: pd.DataFrame())
    monkeypatch.setattr(daily, "_props_for", _fake_props)
    monkeypatch.setattr(daily, "REPORTS_DAILY", tmp_path / "daily")
    daily.cmd_pre_day("2026-07-02")
    n1 = ft.summary()["total"]
    daily.cmd_pre_day("2026-07-02")  # re-correr NO duplica (UNIQUE)
    assert ft.summary()["total"] == n1


def test_pre_kickoff_respects_window(tmp_db, monkeypatch, capsys):
    calls = {"n": 0}

    def fake_get(key, path, params, name, **kw):
        calls["n"] += 1
        return {"response": [_fx()]}  # kickoff 2026-07-02: lejos de ahora

    monkeypatch.setattr(daily, "_get_cached", fake_get)
    daily.cmd_pre_kickoff(999)
    out = capsys.readouterr().out
    assert "Fuera de la ventana" in out
    assert calls["n"] == 1  # solo la consulta del fixture; NO pidió lineups


def test_settle_team_fixture_math(tmp_db):
    ft.log_prediction(fixture_id=5, match="A vs B", market="team_goals_ou_2.5",
                      player_id=0, player_name="-", pred_prob=0.62, line=2.5)
    ft.log_prediction(fixture_id=5, match="A vs B", market="team_1x2_home",
                      player_id=0, player_name="-", pred_prob=0.5)
    ft.log_prediction(fixture_id=5, match="A vs B", market="team_se_clasifica_away",
                      player_id=0, player_name="-", pred_prob=0.4)
    n = ft.settle_team_fixture(5, {
        "goals_total": 3, "corners_total": 11, "yellows_total": 4,
        "shots_total": 22, "sot_total": 9, "result": "home", "btts": 1.0,
        "advanced": "home",
    })
    assert n == 3
    with ft._connect() as conn:  # noqa: SLF001
        rows = {r[0]: r for r in conn.execute(
            "SELECT market, actual, brier FROM props_log WHERE fixture_id=5")}
    assert rows["team_goals_ou_2.5"][1] == 3.0                      # actual = total
    assert rows["team_goals_ou_2.5"][2] == pytest.approx((0.62 - 1.0) ** 2)  # over pegó
    assert rows["team_1x2_home"][2] == pytest.approx((0.5 - 1.0) ** 2)
    assert rows["team_se_clasifica_away"][2] == pytest.approx((0.4 - 0.0) ** 2)


def test_add_odds_updates_row(tmp_db, capsys):
    ft.log_prediction(fixture_id=7, match="A vs B", market="shots",
                      player_id=10, player_name="X", pred_mean=1.5)
    daily.cmd_add_odds(7, 10, "shots", 1.5, 1.85)
    with ft._connect() as conn:  # noqa: SLF001
        line, odds, book = conn.execute(
            "SELECT line, odds, book FROM props_log WHERE fixture_id=7").fetchone()
    assert (line, odds, book) == (1.5, 1.85, "bet365")
    daily.cmd_add_odds(7, 99, "shots", None, 2.0)  # inexistente
    assert "No encontré" in capsys.readouterr().out


def test_unknown_market_still_rejected(tmp_db):
    with pytest.raises(ValueError):
        ft.log_prediction(fixture_id=1, match="A vs B", market="inventado",
                          player_id=0, player_name="-")


def test_fake_engine_pmfs_valid():
    p = _FakeEngine().predict_match("A", "B", when=pd.Timestamp("2026-07-02"))
    for q, pmf in p["pmfs"].items():
        assert isinstance(pmf, np.ndarray) and pmf.sum() == pytest.approx(1.0, abs=1e-9), q
