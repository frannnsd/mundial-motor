"""Tests de los jobs MLB + API /mlb (sin red ni Supabase real: store/engine mockeados)."""

from __future__ import annotations

import threading
import types

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mundial_bot.api import mlb_api
from mundial_bot.research.distributions import count_pmf
from mundial_bot.research.mlb import GRID_MLB, QUANTITIES_MLB
from mundial_bot.wc import jobs, mlb_jobs, store

ACCESS_KEY = "clave-mlb-test"
HEADERS = {"X-Access-Key": ACCESS_KEY}


# ---------------------------------------------------------------------------
# Fixtures compartidas
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_store(monkeypatch):
    """Mockea wc.store completo: graba llamadas, respeta la UNIQUE, sin red."""
    calls = types.SimpleNamespace(
        reports=[], preds=[], odds=[], settles=[], seen=set(),
        get_reports_result=[], get_reports_args=[], select_result=[],
        pending_result=[],
    )

    def _log(**kw):
        kw.setdefault("sport", "wc")
        key = (kw["fixture_id"], kw.get("player_id", 0), kw["market"])
        if key in calls.seen:  # UNIQUE (fixture_id,player_id,market): la primera gana
            return False
        calls.seen.add(key)
        calls.preds.append(kw)
        return True

    monkeypatch.setattr(store, "is_configured", lambda: True)
    monkeypatch.setattr(store, "job_start", lambda job: 1)
    monkeypatch.setattr(store, "job_finish", lambda *a, **k: None)
    monkeypatch.setattr(store, "save_daily_report", lambda row: calls.reports.append(row))
    monkeypatch.setattr(store, "ft_log_prediction", _log)
    monkeypatch.setattr(store, "ft_attach_odds",
                        lambda *a, **kw: (calls.odds.append((a, kw)), 1)[1])
    monkeypatch.setattr(store, "ft_pending", lambda fid: calls.pending_result)
    monkeypatch.setattr(store, "ft_settle_rows",
                        lambda rows: (calls.settles.extend(rows), len(rows))[1])
    monkeypatch.setattr(store, "get_reports",
                        lambda d, sport="wc": (calls.get_reports_args.append((d, sport)),
                                               calls.get_reports_result)[1])
    monkeypatch.setattr(store, "get_report", lambda fid, sport="wc": None)
    monkeypatch.setattr(store, "select", lambda *a, **k: calls.select_result)
    monkeypatch.setattr(store, "daily_backup", lambda: 0)
    monkeypatch.setattr(store, "latest_job_runs", lambda: [])
    return calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_KEY", ACCESS_KEY)
    app = FastAPI()
    app.include_router(mlb_api.router)
    return TestClient(app)


def _game(pk=800123, state="S"):
    return {
        "gamePk": pk,
        "gameDate": "2026-07-02T23:10:00Z",
        "season": "2026",
        "status": {"codedGameState": state},
        "venue": {"name": "Yankee Stadium"},
        "teams": {
            "home": {"team": {"name": "New York Yankees"},
                     "probablePitcher": {"id": 101, "fullName": "Gerrit Cole"}},
            "away": {"team": {"name": "Boston Red Sox"},
                     "probablePitcher": {"id": 202, "fullName": "Brayan Bello"}},
        },
    }


class _FakeMlbEngine:
    """Predicción sintética coherente con el contrato de MlbLiveEngine."""

    def __init__(self):
        self.history = pd.DataFrame([{
            "date": pd.Timestamp("2026-06-30"), "game_pk": 700001,
            "home_team": "New York Yankees", "away_team": "Boston Red Sox",
            "venue": "Yankee Stadium",
        }])

    def predict_game(self, home, away, venue, starter_h_id, starter_a_id, when):
        means = {"runs_h": 4.8, "runs_a": 4.1, "hits_h": 8.6, "hits_a": 7.9,
                 "runs_f5_h": 2.6, "runs_f5_a": 2.2}
        pmfs = {q: count_pmf(means[q], means[q] * 1.2, GRID_MLB[q.rsplit("_", 1)[0]])
                for q in QUANTITIES_MLB}
        return {"pmfs": pmfs, "means": means}


def _fake_starter_props(game, as_of):
    return [
        {"player_id": 101, "player_name": "Gerrit Cole", "side": "home",
         "mu_ks": 7.1, "p_over_5.5": 0.71, "p_over_6.5": 0.55},
        {"player_id": 202, "player_name": "Brayan Bello", "side": "away",
         "mu_ks": 5.2, "p_over_5.5": 0.42, "p_over_6.5": 0.28},
    ]


def _fake_batter_props(team, team_hits_mean, hist, as_of):
    base = 300 if team.startswith("New York") else 400  # ids distintos por equipo
    return [{"player_id": base + i, "player_name": f"Bateador {base + i}",
             "batting_order": i + 1, "mu_hits": round(1.4 - 0.1 * i, 4),
             "p_hit_1plus": 0.7, "p_hr": 0.09} for i in range(3)]


@pytest.fixture
def daily_mocks(fake_store, monkeypatch):
    monkeypatch.setattr(mlb_jobs, "_day_schedule", lambda d, force=False: [_game()])
    monkeypatch.setattr(mlb_jobs, "_build_engine", lambda: _FakeMlbEngine())
    monkeypatch.setattr(mlb_jobs, "_starter_props", _fake_starter_props)
    monkeypatch.setattr(mlb_jobs, "_batter_props", _fake_batter_props)
    return fake_store


# ---------------------------------------------------------------------------
# run_mlb_daily
# ---------------------------------------------------------------------------

def test_run_mlb_daily_guarda_payload_sport_mlb_y_loguea(daily_mocks):
    out = mlb_jobs.run_mlb_daily("2026-07-02")

    assert out["status"] == "ok" and out["matches"] == 1
    assert len(daily_mocks.reports) == 1
    row = daily_mocks.reports[0]
    assert row["sport"] == "mlb"
    assert row["fixture_id"] == 800123
    assert row["report_date"] == "2026-07-02"
    payload = row["payload"]
    assert payload["starters"]["home"]["name"] == "Gerrit Cole"
    pmf = payload["pmfs"]["runs_h"]
    assert isinstance(pmf, list) and sum(pmf) == pytest.approx(1.0, abs=1e-3)
    assert payload["markets"]["moneyline"]["home"] > 0
    assert payload["markets"]["totales"]["8.5"]["over"] > 0  # claves JSON string
    assert payload["props"]["pitchers"][0]["player_name"] == "Gerrit Cole"
    assert payload["props"]["batters"]["home"][0]["mu_hits"] > 0

    # ≥8 predicciones por juego, todas sport='mlb' y con as_of en notes.
    assert out["predictions"] >= 8
    assert all(p["sport"] == "mlb" for p in daily_mocks.preds)
    assert all(p["notes"].startswith("as_of=") for p in daily_mocks.preds)
    markets = {p["market"] for p in daily_mocks.preds}
    assert {"mlb_ml_home", "mlb_ml_away", "mlb_total_ou_8.5", "mlb_f5_ou_4.5",
            "mlb_rl_home_1.5", "ks", "mlb_hits", "mlb_hr"} <= markets
    ks = next(p for p in daily_mocks.preds if p["market"] == "ks")
    assert ks["player_id"] in (101, 202) and ks["line"] == 5.5
    assert ks["pred_mean"] > 0 and 0 < ks["pred_prob"] < 1


def test_run_mlb_daily_es_idempotente(daily_mocks):
    out1 = mlb_jobs.run_mlb_daily("2026-07-02")
    n_after_first = len(daily_mocks.preds)
    out2 = mlb_jobs.run_mlb_daily("2026-07-02")

    assert out1["predictions"] == n_after_first
    assert out2["status"] == "ok" and out2["predictions"] == 0  # UNIQUE: primera gana
    assert len(daily_mocks.preds) == n_after_first


def test_run_mlb_daily_sin_store_no_crashea(monkeypatch):
    monkeypatch.setattr(store, "is_configured", lambda: False)
    out = mlb_jobs.run_mlb_daily("2026-07-02")
    assert out["status"] == "skipped"


# ---------------------------------------------------------------------------
# _mlb_team_actual y liquidación
# ---------------------------------------------------------------------------

def test_mlb_team_actual_a_mano():
    actuals = {"runs_h": 5, "runs_a": 3, "total": 8, "f5_total": 5,
               "margin": 2, "winner": "home"}
    assert mlb_jobs._mlb_team_actual("mlb_ml_home", None, actuals) == (1.0, 1.0)
    assert mlb_jobs._mlb_team_actual("mlb_ml_away", None, actuals) == (0.0, 0.0)
    assert mlb_jobs._mlb_team_actual("mlb_total_ou_8.5", 8.5, actuals) == (8.0, 0.0)
    assert mlb_jobs._mlb_team_actual("mlb_total_ou_8.5", 7.5, actuals) == (8.0, 1.0)
    assert mlb_jobs._mlb_team_actual("mlb_f5_ou_4.5", 4.5, actuals) == (5.0, 1.0)
    assert mlb_jobs._mlb_team_actual("mlb_rl_home_1.5", 1.5, actuals) == (2.0, 1.0)
    assert mlb_jobs._mlb_team_actual("mlb_total_ou_8.5", None, actuals) is None
    assert mlb_jobs._mlb_team_actual("otra_cosa", None, actuals) is None


def _pending_row(**kw):
    base = {"id": 1, "sport": "mlb", "fixture_id": 800123, "market": "mlb_ml_home",
            "player_id": 0, "pred_mean": None, "pred_prob": None, "line": None}
    base.update(kw)
    return base


def test_run_mlb_settle_liquida_equipo_y_props(fake_store, monkeypatch):
    final = _game(state="F")
    # linescore: home 5 (2 en F5), away 3 (1 en F5) → total 8, f5 3, margen 2.
    inn = [{"home": {"runs": h}, "away": {"runs": a}}
           for h, a in ((1, 0), (0, 1), (1, 0), (0, 0), (0, 0),
                        (1, 0), (0, 1), (2, 0), (0, 1))]
    final["linescore"] = {"innings": inn,
                          "teams": {"home": {"runs": 5, "hits": 10},
                                    "away": {"runs": 3, "hits": 7}}}
    monkeypatch.setattr(mlb_jobs, "_day_schedule", lambda d, force=False: [final])
    monkeypatch.setattr(mlb_jobs, "_boxscore_stats", lambda pk: {
        101: {"pitched": True, "batted": False, "strikeouts": 7,
              "hits": 0, "home_runs": 0},
        303: {"pitched": False, "batted": True, "strikeouts": 0,
              "hits": 2, "home_runs": 0},
    })
    fake_store.pending_result = [
        _pending_row(id=1, market="mlb_ml_home", pred_prob=0.55),
        _pending_row(id=2, market="mlb_total_ou_8.5", pred_prob=0.6, line=8.5),
        _pending_row(id=3, market="ks", player_id=101, pred_mean=7.1,
                     pred_prob=0.71, line=5.5),
        _pending_row(id=4, market="mlb_hits", player_id=303, pred_mean=1.4),
        _pending_row(id=5, market="mlb_hr", player_id=303, pred_prob=0.09),
        _pending_row(id=6, market="mlb_hr", player_id=999),   # no jugó: pendiente
        _pending_row(id=7, market="anota", sport="wc", player_id=9),  # WC: se ignora
    ]

    out = mlb_jobs.run_mlb_settle("2026-07-02")

    assert out["status"] == "ok" and out["fixtures"] == 1
    settled = {s["id"]: s for s in fake_store.settles}
    assert set(settled) == {1, 2, 3, 4, 5}
    assert settled[1]["actual"] == 1.0                          # ganó el local
    assert settled[1]["brier"] == pytest.approx((0.55 - 1.0) ** 2)
    assert settled[2]["actual"] == 8.0                          # 8 ≤ 8.5 → under
    assert settled[2]["brier"] == pytest.approx(0.6 ** 2)
    assert settled[3]["actual"] == 7.0                          # 7 Ks > 5.5 → over
    assert settled[3]["brier"] == pytest.approx((0.71 - 1.0) ** 2)
    assert settled[4]["actual"] == 2.0 and settled[4]["brier"] is None
    assert settled[5]["actual"] == 0.0                          # sin HR
    assert settled[5]["brier"] == pytest.approx(0.09 ** 2)


# ---------------------------------------------------------------------------
# API /mlb
# ---------------------------------------------------------------------------

def test_api_requiere_access_key(client, fake_store):
    assert client.get("/mlb/today").status_code == 401
    assert client.get("/mlb/today", headers={"X-Access-Key": "otra"}).status_code == 401
    r = client.get("/mlb/today", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["matches"] == []


def test_today_lee_sport_mlb_y_arma_cards(client, fake_store):
    fake_store.get_reports_result = [{
        "fixture_id": 800123, "sport": "mlb",
        "kickoff_utc": "2026-07-02T23:10:00+00:00",
        "home": "New York Yankees", "away": "Boston Red Sox",
        "payload": {
            "venue": "Yankee Stadium",
            "starters": {"home": {"id": 101, "name": "Gerrit Cole"},
                         "away": {"id": 202, "name": "Brayan Bello"}},
            "means": {"runs_h": 4.812345, "runs_a": 4.1},
            "markets": {
                "moneyline": {"home": 0.56, "away": 0.44},
                "totales": {"8.5": {"over": 0.52, "under": 0.48}},
                "run_line": {"home_-1.5": 0.35, "away_+1.5": 0.65},
                "f5": {"totales": {"4.5": {"over": 0.49}}},
            },
        },
    }]
    r = client.get("/mlb/today?date=2026-07-02", headers=HEADERS)
    assert r.status_code == 200
    assert fake_store.get_reports_args[-1] == ("2026-07-02", "mlb")
    card = r.json()["matches"][0]
    assert card["game_pk"] == 800123
    assert card["starters"]["home"]["name"] == "Gerrit Cole"
    assert card["moneyline"] == {"home": 0.56, "away": 0.44}
    probs = {m["market"]: m["prob"] for m in card["top_markets"]}
    assert probs == {"mlb_total_ou_8.5": 0.52, "mlb_f5_ou_4.5": 0.49,
                     "mlb_rl_home_1.5": 0.35}
    assert card["means_compact"]["runs_h"] == 4.81


def test_today_valida_fecha(client, fake_store):
    assert client.get("/mlb/today?date=ayer", headers=HEADERS).status_code == 422


def test_match_inexistente_404(client, fake_store):
    assert client.get("/mlb/match/12345", headers=HEADERS).status_code == 404


def test_post_odds_manda_sport_mlb(client, fake_store):
    body = {"fixture_id": 800123, "player_id": 101, "market": "ks",
            "line": 5.5, "odds": 1.9}
    r = client.post("/mlb/odds", json=body, headers=HEADERS)
    assert r.status_code == 200 and r.json()["updated"] == 1
    args, kwargs = fake_store.odds[0]
    assert args == (800123, 101, "ks")
    assert kwargs == {"line": 5.5, "odds": 1.9, "stake": None, "sport": "mlb"}


def test_forward_test_filtra_sport_mlb(client, fake_store):
    fake_store.select_result = [
        {"id": 1, "sport": "mlb", "market": "mlb_ml_home", "pred_prob": 0.6,
         "actual": 1.0, "brier": 0.16, "settled_at": "2026-07-02T00:00:00+00:00"},
        {"id": 2, "market": "team_btts", "pred_prob": 0.5, "actual": 1.0,
         "brier": 0.25, "settled_at": "2026-07-02T00:00:00+00:00"},  # wc implícito
    ]
    r = client.get("/mlb/forward-test", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 1  # la fila WC quedó afuera
    assert body["summary"]["brier_mean"] == pytest.approx(0.16)


def test_admin_run_dispara_el_job_en_thread(client, fake_store, monkeypatch):
    ran = threading.Event()
    monkeypatch.setitem(mlb_api._RUNNABLE, "mlb_daily", ran.set)
    r = client.post("/mlb/admin/run/mlb_daily", headers=HEADERS)
    assert r.json() == {"started": True, "job": "mlb_daily"}
    assert ran.wait(timeout=2.0)
    assert client.post("/mlb/admin/run/nada", headers=HEADERS).status_code == 404


# ---------------------------------------------------------------------------
# Compat del store multi-deporte (los callers WC no cambian)
# ---------------------------------------------------------------------------

def test_save_daily_report_default_sport_wc(monkeypatch):
    captured = {}
    monkeypatch.setattr(store, "upsert",
                        lambda table, rows, on_conflict: captured.update(
                            table=table, rows=rows, on_conflict=on_conflict))
    original = {"fixture_id": 1, "report_date": "2026-07-02"}
    store.save_daily_report(original)
    assert captured["on_conflict"] == "sport,fixture_id"
    assert captured["rows"][0]["sport"] == "wc"
    assert "sport" not in original  # no muta la fila del caller


def test_ft_log_prediction_default_sport_wc(monkeypatch):
    captured = {}
    monkeypatch.setattr(store, "insert_ignore",
                        lambda table, rows, on_conflict: (captured.update(
                            rows=rows, on_conflict=on_conflict), 1)[1])
    assert store.ft_log_prediction(fixture_id=1, match="A vs B", market="team_btts")
    assert captured["rows"][0]["sport"] == "wc"
    # el on_conflict NO cambia: los mercados MLB son disjuntos por diseño.
    assert captured["on_conflict"] == "fixture_id,player_id,market"


def test_ft_attach_odds_filtra_por_sport(monkeypatch):
    captured = {}
    monkeypatch.setattr(store, "update",
                        lambda table, filters, patch: (captured.update(
                            filters=filters, patch=patch), 1)[1])
    store.ft_attach_odds(800123, 0, "mlb_ml_home", line=None, odds=1.9, sport="mlb")
    assert captured["filters"]["sport"] == "eq.mlb"
    store.ft_attach_odds(7, 0, "team_btts", line=None, odds=1.8)
    assert captured["filters"]["sport"] == "eq.wc"  # default: comportamiento WC


def test_compute_forward_test_filtra_por_sport():
    rows = [
        {"id": 1, "market": "team_btts", "settled_at": None},          # wc implícito
        {"id": 2, "sport": "mlb", "market": "mlb_ml_home", "settled_at": None},
    ]
    out = jobs.compute_forward_test(rows, sport="mlb")
    assert out["summary"]["total"] == 1
    out_all = jobs.compute_forward_test(rows)  # sin sport: todas (compat)
    assert out_all["summary"]["total"] == 2
