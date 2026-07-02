"""Tests de los jobs cloud + API /wc (sin red ni Supabase real: store mockeado)."""

from __future__ import annotations

import threading
import types

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mundial_bot.api import wc_api
from mundial_bot.research.distributions import count_pmf
from mundial_bot.wc import daily, jobs, store

ACCESS_KEY = "clave-de-test"
HEADERS = {"X-Access-Key": ACCESS_KEY}


# ---------------------------------------------------------------------------
# Fixtures compartidas
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_store(monkeypatch):
    """Mockea wc.store completo: graba llamadas, nunca toca la red."""
    calls = types.SimpleNamespace(
        reports=[], preds=[], odds=[], settles=[], upserts=[],
        get_reports_result=[], select_result=[], pending_result=[],
    )
    monkeypatch.setattr(store, "is_configured", lambda: True)
    monkeypatch.setattr(store, "job_start", lambda job: 1)
    monkeypatch.setattr(store, "job_finish", lambda *a, **k: None)
    monkeypatch.setattr(store, "save_daily_report", lambda row: calls.reports.append(row))
    monkeypatch.setattr(store, "ft_log_prediction",
                        lambda **kw: (calls.preds.append(kw), True)[1])
    monkeypatch.setattr(store, "ft_attach_odds",
                        lambda *a, **kw: (calls.odds.append((a, kw)), 1)[1])
    monkeypatch.setattr(store, "ft_pending", lambda fid: calls.pending_result)
    monkeypatch.setattr(store, "ft_settle_rows",
                        lambda rows: (calls.settles.extend(rows), len(rows))[1])
    monkeypatch.setattr(store, "get_reports", lambda d: calls.get_reports_result)
    monkeypatch.setattr(store, "get_report", lambda fid: None)
    monkeypatch.setattr(store, "select", lambda *a, **k: calls.select_result)
    monkeypatch.setattr(store, "upsert",
                        lambda table, rows, on_conflict: calls.upserts.append((table, rows)))
    monkeypatch.setattr(store, "daily_backup", lambda: 0)
    monkeypatch.setattr(store, "latest_job_runs", lambda: [])
    return calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_KEY", ACCESS_KEY)
    app = FastAPI()
    app.include_router(wc_api.router)
    return TestClient(app)


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


# ---------------------------------------------------------------------------
# Auth de la API
# ---------------------------------------------------------------------------

def test_api_requiere_access_key(client, fake_store):
    assert client.get("/wc/today").status_code == 401
    assert client.get("/wc/today", headers={"X-Access-Key": "otra"}).status_code == 401
    r = client.get("/wc/today", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["matches"] == []


def test_api_sin_web_access_key_devuelve_503(monkeypatch, fake_store):
    monkeypatch.delenv("WEB_ACCESS_KEY", raising=False)
    app = FastAPI()
    app.include_router(wc_api.router)
    c = TestClient(app)
    r = c.get("/wc/today", headers={"X-Access-Key": "cualquiera"})
    assert r.status_code == 503
    assert "WEB_ACCESS_KEY" in r.json()["detail"]


# ---------------------------------------------------------------------------
# run_daily
# ---------------------------------------------------------------------------

def test_run_daily_guarda_payload_y_loguea(fake_store, monkeypatch):
    monkeypatch.setattr(daily, "_day_fixtures", lambda *a, **k: [_fx()])
    monkeypatch.setattr(daily, "_build_engine", lambda: _FakeEngine())
    monkeypatch.setattr(daily, "_player_table", lambda: pd.DataFrame())
    monkeypatch.setattr(daily, "_props_for", _fake_props)

    out = jobs.run_daily("2026-07-02")

    assert out["status"] == "ok" and out["matches"] == 1
    assert len(fake_store.reports) == 1
    row = fake_store.reports[0]
    assert row["fixture_id"] == 999
    assert row["report_date"] == "2026-07-02"
    assert row["is_knockout"] is True and row["xi_confirmed"] is False
    payload = row["payload"]
    pmf = payload["pmfs"]["goals_h"]
    assert isinstance(pmf, list) and sum(pmf) == pytest.approx(1.0, abs=1e-3)
    assert payload["markets90"]["goles_ou"]["2.5"]["over"] > 0  # claves JSON string
    assert payload["knockout"]["se_clasifica"]["home"] > 0
    assert payload["props"]["home"][0]["player_name"] == "Balogun"
    # Predicciones: mismas familias que el pre-day local, con as_of en notes.
    assert len(fake_store.preds) >= 10
    markets = {p["market"] for p in fake_store.preds}
    assert {"team_1x2_home", "team_btts", "team_goals_ou_2.5",
            "team_se_clasifica_home", "shots", "anota"} <= markets
    assert all(p["notes"].startswith("as_of=") for p in fake_store.preds)


def test_run_daily_sin_store_no_crashea(monkeypatch):
    monkeypatch.setattr(store, "is_configured", lambda: False)
    out = jobs.run_daily("2026-07-02")
    assert out["status"] == "skipped"


# ---------------------------------------------------------------------------
# run_lineups
# ---------------------------------------------------------------------------

def test_run_lineups_fuera_de_ventana_no_llama(fake_store, monkeypatch):
    fake_store.get_reports_result = [{
        "fixture_id": 999, "kickoff_utc": "2026-12-01T20:00:00+00:00",
        "home": "A", "away": "B", "is_knockout": False,
        "xi_confirmed": False, "payload": {},
    }]
    lineup_calls = {"n": 0}
    monkeypatch.setattr(
        daily, "_get_cached",
        lambda *a, **k: (lineup_calls.__setitem__("n", lineup_calls["n"] + 1), {})[1],
    )
    monkeypatch.setattr(jobs, "_last_lineup_try", {})

    out = jobs.run_lineups()

    assert out["status"] == "ok"
    assert out["in_window"] == 0 and out["confirmed"] == 0
    assert lineup_calls["n"] == 0  # cero llamadas de lineups fuera de la ventana


def test_run_lineups_respeta_rate_limit_por_fixture(fake_store, monkeypatch):
    kickoff = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=30)).isoformat()
    fake_store.get_reports_result = [{
        "fixture_id": 555, "kickoff_utc": kickoff,
        "home": "A", "away": "B", "is_knockout": False,
        "xi_confirmed": False, "payload": {},
    }]
    lineup_calls = {"n": 0}
    monkeypatch.setattr(
        daily, "_get_cached",
        lambda *a, **k: (lineup_calls.__setitem__("n", lineup_calls["n"] + 1),
                         {"response": []})[1],  # XI todavía no publicado
    )
    monkeypatch.setattr(jobs, "_last_lineup_try", {})

    out1 = jobs.run_lineups()
    out2 = jobs.run_lineups()  # inmediato: el rate-limit 1/5min lo frena

    assert out1["attempts"] == 1 and lineup_calls["n"] == 1
    assert out2["attempts"] == 0 and lineup_calls["n"] == 1


# ---------------------------------------------------------------------------
# POST /wc/odds
# ---------------------------------------------------------------------------

def test_post_odds_llama_attach_con_args_correctos(client, fake_store):
    body = {"fixture_id": 7, "player_id": 10, "market": "shots",
            "line": 1.5, "odds": 1.85, "stake": 2.0}
    r = client.post("/wc/odds", json=body, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["updated"] == 1
    args, kwargs = fake_store.odds[0]
    assert args == (7, 10, "shots")
    assert kwargs == {"line": 1.5, "odds": 1.85, "stake": 2.0}


def test_post_odds_rechaza_cuota_invalida(client, fake_store):
    body = {"fixture_id": 7, "market": "shots", "odds": 0.9}
    assert client.post("/wc/odds", json=body, headers=HEADERS).status_code == 422
    assert fake_store.odds == []


# ---------------------------------------------------------------------------
# Forward-test (cálculo compartido job weekly / GET /wc/forward-test)
# ---------------------------------------------------------------------------

def _row(**kw):
    base = {"id": 1, "fixture_id": 1, "market": "team_1x2_home", "player_id": 0,
            "player_name": "-", "pred_mean": None, "pred_prob": None, "line": None,
            "odds": None, "actual": None, "brier": None, "settled_at": None}
    base.update(kw)
    return base


def test_forward_test_roi_y_ev_a_mano():
    rows = [
        # ganó el home con cuota 2.0 → ROI +1.0 ; EV = 0.6·2.0−1 = +0.2
        _row(id=1, market="team_1x2_home", pred_prob=0.6, odds=2.0,
             actual=1.0, brier=0.16, settled_at="2026-07-01T00:00:00+00:00"),
        # 2 goles ≤ 2.5 → no pegó, ROI −1.0 ; EV = 0.7·1.9−1 = +0.33
        _row(id=2, market="team_goals_ou_2.5", pred_prob=0.7, line=2.5, odds=1.9,
             actual=2.0, brier=0.49, settled_at="2026-07-01T00:00:00+00:00"),
        # conteo con media (sin prob): solo aporta al MAE
        _row(id=3, market="shots", player_id=9, pred_mean=2.0, actual=3.0,
             settled_at="2026-07-01T00:00:00+00:00"),
        # pendiente sin cuota
        _row(id=4, market="anota", player_id=9, pred_prob=0.3),
    ]
    out = jobs.compute_forward_test(rows)

    s = out["summary"]
    assert (s["total"], s["settled"], s["pending"]) == (4, 3, 1)
    assert s["brier_mean"] == pytest.approx((0.16 + 0.49) / 2)
    assert s["mae"] == pytest.approx(1.0)

    ev = out["ev"]
    assert ev["n_con_cuota"] == 2 and ev["n_liquidadas_con_cuota"] == 2
    assert ev["roi_stake_plano"] == pytest.approx((1.0 - 1.0) / 2)     # = 0.0
    assert ev["ev_teorico_medio"] == pytest.approx((0.2 + 0.33) / 2)   # = 0.265

    by = {m["market"]: m for m in out["by_market"]}
    assert by["team_1x2_home"]["pred_avg"] == pytest.approx(0.6)
    assert by["team_1x2_home"]["real_avg"] == pytest.approx(1.0)
    assert by["team_1x2_home"]["gap"] == pytest.approx(-0.4)
    assert by["team_goals_ou_2.5"]["real_avg"] == pytest.approx(0.0)  # 2 < 2.5: under
    assert by["shots"]["n_settled"] == 1 and by["shots"]["pred_avg"] is None
    assert out["worst_markets"] == []  # nadie llega a n_settled >= 10


def test_forward_test_endpoint_usa_props_log(client, fake_store):
    fake_store.select_result = [
        _row(id=1, market="team_btts", pred_prob=0.5, actual=1.0, brier=0.25,
             settled_at="2026-07-01T00:00:00+00:00"),
    ]
    r = client.get("/wc/forward-test", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 1 and body["summary"]["settled"] == 1
    assert body["summary"]["brier_mean"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# GET /wc/today y /wc/match — contrato de la web
# ---------------------------------------------------------------------------

def test_today_arma_las_cards(client, fake_store):
    fake_store.get_reports_result = [{
        "fixture_id": 5, "kickoff_utc": "2026-07-02T20:00:00+00:00",
        "home": "Argentina", "away": "Brazil", "round": "Round of 16",
        "is_knockout": True, "xi_confirmed": False,
        "payload": {
            "means": {"goals_h": 1.61234, "goals_a": 1.1},
            "markets90": {
                "1x2": {"home": 0.5, "draw": 0.3, "away": 0.2},
                "btts": {"yes": 0.55, "no": 0.45},
                "goles_ou": {"2.5": {"over": 0.61, "under": 0.39}},
                "corners_ou": {"9.5": {"over": 0.52}},
                "tarjetas_ou": {"3.5": {"over": 0.47}},
            },
            "knockout": {"se_clasifica": {"home": 0.62, "away": 0.38}},
        },
    }]
    r = client.get("/wc/today?date=2026-07-02", headers=HEADERS)
    assert r.status_code == 200
    card = r.json()["matches"][0]
    assert card["one_x_two"] == {"home": 0.5, "draw": 0.3, "away": 0.2}
    assert card["se_clasifica"]["home"] == 0.62
    probs = {m["market"]: m["prob"] for m in card["top_markets"]}
    assert probs == {"goles_ou_2.5": 0.61, "corners_ou_9.5": 0.52,
                     "tarjetas_ou_3.5": 0.47, "btts": 0.55}
    assert card["means_compact"]["goals_h"] == 1.61


def test_today_valida_fecha(client, fake_store):
    assert client.get("/wc/today?date=ayer", headers=HEADERS).status_code == 422


def test_match_inexistente_404(client, fake_store):
    assert client.get("/wc/match/12345", headers=HEADERS).status_code == 404


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def test_admin_run_dispara_el_job_en_thread(client, fake_store, monkeypatch):
    ran = threading.Event()
    monkeypatch.setitem(wc_api._RUNNABLE, "daily", ran.set)
    r = client.post("/wc/admin/run/daily", headers=HEADERS)
    assert r.json() == {"started": True, "job": "daily"}
    assert ran.wait(timeout=2.0)
    assert client.post("/wc/admin/run/nada", headers=HEADERS).status_code == 404


def test_admin_backup_descargable(client, fake_store):
    fake_store.select_result = [_row(id=1)]
    r = client.get("/wc/admin/backup", headers=HEADERS)
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert r.json()[0]["id"] == 1
