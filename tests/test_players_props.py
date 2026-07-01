"""Tests de la capa de props por jugador (Fase B) — sin red, fixtures sintéticos."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.collectors.players_wc import parse_fixture_players
from mundial_bot.forward_test import log as ft_log
from mundial_bot.players.props import expected_minutes, match_props
from mundial_bot.players.shares import player_shares, position_rates

# ---------------------------------------------------------------------------
# Helpers sintéticos
# ---------------------------------------------------------------------------


def _row(fid, team, pid, name, pos, minutes, *, sub=False, **stats):
    base = {
        "fixture_id": fid, "date": "2026-06-15", "team": team,
        "player_id": pid, "player_name": name, "position": pos,
        "substitute": sub, "minutes": minutes,
        "shots": 0, "sot": 0, "goals": 0, "assists": 0,
        "fouls_committed": 0, "fouls_drawn": 0, "yellow": 0, "red": 0, "tackles": 0,
    }
    base.update(stats)
    return base


def _synthetic_table() -> pd.DataFrame:
    """Torneo chico: 'Testland' (equipo bajo test) + pool de relleno por puesto.

    El pool (equipo 'Relleno') fija los promedios por puesto del shrinkage:
    mediocampistas con ~1 remate por-90, defensores con ~0.3, etc.
    """
    rows = []
    # Pool: 3 fixtures × 5 jugadores por puesto, 90' cada uno.
    pool_stats = {"G": {}, "D": {"shots": 0, "tackles": 3},
                  "M": {"shots": 1, "fouls_committed": 1}, "F": {"shots": 3, "goals": 1}}
    pid = 1000
    for pos, stats in pool_stats.items():
        for j in range(5):
            pid += 1
            for fid in (101, 102, 103):
                rows.append(_row(fid, "Relleno", pid, f"pool_{pos}{j}", pos, 90, **stats))
    # Testland: 11 titulares (90'), 1 suplente fijo (20'), 1 citado sin minutos.
    for fid in (201, 202, 203):
        rows.append(_row(fid, "Testland", 1, "arquero", "G", 90))
        for k in range(2, 6):
            rows.append(_row(fid, "Testland", k, f"def{k}", "D", 90, tackles=3))
        for k in range(6, 9):
            rows.append(_row(fid, "Testland", k, f"med{k}", "M", 90, shots=1,
                             fouls_committed=2, yellow=(1 if k == 6 else 0)))
        for k in range(9, 12):
            rows.append(_row(fid, "Testland", k, f"del{k}", "F", 90, shots=4,
                             sot=2, goals=1))
        rows.append(_row(fid, "Testland", 12, "suplente", "M", 20, sub=True, shots=1))
        rows.append(_row(fid, "Testland", 13, "banco", "M", 0, sub=True))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1) Parseo: None → 0
# ---------------------------------------------------------------------------


def test_parse_none_es_cero():
    raw = {
        "response": [{
            "team": {"name": "Argentina"},
            "players": [{
                "player": {"id": 7, "name": "Julián"},
                "statistics": [{
                    "games": {"minutes": None, "position": "F", "substitute": True},
                    "shots": {"total": None, "on": None},
                    "goals": {"total": None, "assists": None},
                    "fouls": {"committed": None, "drawn": 2},
                    "cards": {"yellow": None, "red": None},
                    "tackles": {"total": None},
                }],
            }],
        }],
    }
    rows = parse_fixture_players(raw, fixture_id=999, date="2026-06-20")
    assert len(rows) == 1
    r = rows[0]
    assert r["minutes"] == 0 and r["shots"] == 0 and r["sot"] == 0
    assert r["goals"] == 0 and r["yellow"] == 0 and r["tackles"] == 0
    assert r["fouls_drawn"] == 2                 # los valores reales se conservan
    assert r["substitute"] is True and r["position"] == "F"


# ---------------------------------------------------------------------------
# 2) Shrinkage hacia el promedio del puesto
# ---------------------------------------------------------------------------


def test_shrinkage_pocos_minutos_va_al_puesto():
    table = _synthetic_table()
    # Jugador M con 30' y tasa cruda altísima (3 remates en 30' = 9 por-90).
    extra = pd.DataFrame([_row(204, "Testland", 50, "rafaga", "M", 30, sub=True, shots=3)])
    table = pd.concat([table, extra], ignore_index=True)

    shares = player_shares(table, "Testland")
    pos_rate = float(position_rates(table).loc["M", "shots"])
    row = shares[shares["player_id"] == 50].iloc[0]
    raw_rate = 90.0 * 3 / 30.0
    # Con 30' domina el prior del puesto: queda mucho más cerca del puesto que de su tasa.
    assert abs(row["rate_shots"] - pos_rate) < abs(row["rate_shots"] - raw_rate)
    assert row["rate_shots"] < raw_rate / 2


def test_shrinkage_muchos_minutos_conserva_su_tasa():
    table = _synthetic_table()
    # Jugador M con 540' (6×90) y tasa cruda 3.0 por-90 (18 remates), lejos del puesto (~1).
    extra = pd.DataFrame([
        _row(300 + i, "Testland", 60, "tanque", "M", 90, shots=3) for i in range(6)
    ])
    table = pd.concat([table, extra], ignore_index=True)

    shares = player_shares(table, "Testland")
    pos_rate = float(position_rates(table).loc["M", "shots"])
    row = shares[shares["player_id"] == 60].iloc[0]
    raw_rate = 3.0
    assert abs(row["rate_shots"] - raw_rate) < abs(row["rate_shots"] - pos_rate)


# ---------------------------------------------------------------------------
# 3) COHERENCIA: Σ medias de jugadores == media del equipo
# ---------------------------------------------------------------------------

TEAM_TOTALS = {
    "shots": (13.0, 16.0), "sot": (4.5, 5.0), "goals": (1.6, 1.7),
    "yellow": (2.2, 2.4), "fouls_committed": (11.0, 12.0),
}


@pytest.mark.parametrize("lineup", [None, set(range(1, 12)), {1, 2, 3, 12, 13}])
def test_coherencia_suma_igual_al_total_del_equipo(lineup):
    shares = player_shares(_synthetic_table(), "Testland")
    props = match_props(TEAM_TOTALS, shares, lineup_confirmed=lineup)
    for stat, (mean, _var) in TEAM_TOTALS.items():
        assert float(props[f"mu_{stat}"].sum()) == pytest.approx(mean, abs=1e-6)


def test_coherencia_horizon_120():
    shares = player_shares(_synthetic_table(), "Testland")
    props = match_props(TEAM_TOTALS, shares, horizon="120")
    for stat, (mean, _var) in TEAM_TOTALS.items():
        assert float(props[f"mu_{stat}"].sum()) == pytest.approx(mean, abs=1e-6)


def test_props_probs_derivadas_en_rango():
    shares = player_shares(_synthetic_table(), "Testland")
    props = match_props(TEAM_TOTALS, shares)
    for col in ("p_scores", "p_card", "p_shots_2plus"):
        assert col in props.columns
        assert ((props[col] >= 0) & (props[col] < 1)).all()
    # Un delantero titular remata más que el arquero.
    fw = props[props["player_id"] == 9].iloc[0]
    gk = props[props["player_id"] == 1].iloc[0]
    assert fw["mu_shots"] > gk["mu_shots"]


def test_match_props_stat_desconocido_falla_claro():
    shares = player_shares(_synthetic_table(), "Testland")
    with pytest.raises(ValueError, match="sin rate"):
        match_props({"corners": (5.0, 5.0)}, shares)


# ---------------------------------------------------------------------------
# 4) Minutos esperados y leakage del XI
# ---------------------------------------------------------------------------

_ROTADO = {
    # Titular habitual pero con rotación: citado 3 veces, jugó 2 de titular.
    "player_id": 8, "min_avg": (90 + 85 + 0) / 3, "min_avg_starter": 87.5, "min_avg_sub": 0.0,
}


def test_expected_minutes_confirmado_mas_que_probable_mas_que_afuera():
    probable = expected_minutes(_ROTADO)                                # sin lineup
    confirmado = expected_minutes(_ROTADO, lineup_confirmed={8})        # en el XI
    afuera = expected_minutes(_ROTADO, lineup_confirmed={99})           # fuera del XI
    assert afuera < probable < confirmado
    assert confirmado == pytest.approx(87.5)       # min(90, media como titular)
    assert afuera == pytest.approx(15.0)           # nunca entró de suplente → default


def test_expected_minutes_horizon_120_escala():
    assert expected_minutes(_ROTADO, horizon="120") == pytest.approx(
        expected_minutes(_ROTADO) * 120 / 90
    )
    with pytest.raises(ValueError, match="horizon"):
        expected_minutes(_ROTADO, horizon="45")


# ---------------------------------------------------------------------------
# 5) Forward test: log + settle + idempotencia (sin red: fetch mockeado)
# ---------------------------------------------------------------------------

_REAL_RAW = {
    "response": [{
        "team": {"name": "Argentina"},
        "players": [
            {"player": {"id": 10, "name": "Messi"},
             "statistics": [{"games": {"minutes": 90, "position": "F", "substitute": False},
                             "shots": {"total": 4, "on": 2}, "goals": {"total": 1, "assists": None},
                             "fouls": {"committed": 1, "drawn": 3},
                             "cards": {"yellow": None, "red": None}, "tackles": {"total": None}}]},
            {"player": {"id": 20, "name": "De Paul"},
             "statistics": [{"games": {"minutes": 90, "position": "M", "substitute": False},
                             "shots": {"total": 1, "on": None}, "goals": {"total": None},
                             "fouls": {"committed": 3, "drawn": 1},
                             "cards": {"yellow": 1, "red": None}, "tackles": {"total": 4}}]},
        ],
    }],
}


def test_forward_test_log_settle_summary(tmp_path, monkeypatch):
    db = tmp_path / "ft.sqlite"
    monkeypatch.setattr(ft_log, "fetch_fixture_players", lambda key, fid: _REAL_RAW)

    assert ft_log.log_prediction(
        fixture_id=555, match="Argentina vs X", market="shots",
        player_id=10, player_name="Messi", pred_mean=3.1, pred_prob=0.70, line=2.5,
        odds=1.9, book="bet365", db_path=db,
    )
    assert ft_log.log_prediction(
        fixture_id=555, match="Argentina vs X", market="anota",
        player_id=10, player_name="Messi", pred_prob=0.55, db_path=db,
    )
    assert ft_log.log_prediction(
        fixture_id=555, match="Argentina vs X", market="tarjeta",
        player_id=20, player_name="De Paul", pred_prob=0.30, db_path=db,
    )
    # Idempotencia del log: re-insertar la misma (fixture, player, market) no duplica.
    assert not ft_log.log_prediction(
        fixture_id=555, match="Argentina vs X", market="shots",
        player_id=10, player_name="Messi", pred_mean=9.9, db_path=db,
    )

    assert ft_log.settle_fixture("fake-key", 555, db_path=db) == 3
    assert ft_log.settle_fixture("fake-key", 555, db_path=db) == 0   # idempotente

    s = ft_log.summary(db_path=db)
    assert s["total"] == 3 and s["settled"] == 3 and s["pending"] == 0
    # Messi: 4 > 2.5 → over gana → brier del over = (0.70-1)²; anota=1 → (0.55-1)².
    # De Paul: amarilla → (0.30-1)². Promedio de los tres.
    esperado = ((0.70 - 1) ** 2 + (0.55 - 1) ** 2 + (0.30 - 1) ** 2) / 3
    assert s["brier_mean"] == pytest.approx(esperado)
    assert s["mae_mean"] == pytest.approx(abs(3.1 - 4))   # única fila con pred_mean


def test_forward_test_mercado_desconocido_falla():
    with pytest.raises(ValueError, match="desconocido"):
        ft_log.log_prediction(
            fixture_id=1, match="A vs B", market="dribbles",
            player_id=1, player_name="x", db_path=None,
        )
