"""Tests del ledger SQLite (Agente 6)."""

from __future__ import annotations

import pytest

from mundial_bot.ledger.store import Ledger


@pytest.fixture()
def ledger():
    lg = Ledger(":memory:")
    yield lg
    lg.close()


def _record(lg, *, match="A vs B", odds=2.0, stake=2.5, edge=0.10):
    return lg.record(
        created_at="2026-06-14", match=match, market="1X2", selection="home",
        odds=odds, model_prob=0.55, edge=edge, stake=stake, bookmaker="Pinnacle",
    )


def test_record_y_open_picks(ledger):
    _record(ledger)
    opens = ledger.open_picks()
    assert len(opens) == 1
    assert opens[0]["match"] == "A vs B"
    assert opens[0]["status"] == "pending"


def test_settle_ganado_calcula_pnl(ledger):
    pid = _record(ledger, odds=2.0, stake=10.0)
    ledger.settle(pid, won=True)
    s = ledger.summary()
    assert s.won == 1
    assert s.returned == pytest.approx(20.0)   # 10 * 2.0
    assert s.roi == pytest.approx(1.0)          # (20-10)/10


def test_settle_perdido(ledger):
    pid = _record(ledger, odds=2.0, stake=10.0)
    ledger.settle(pid, won=False)
    s = ledger.summary()
    assert s.won == 0
    assert s.returned == 0.0
    assert s.roi == pytest.approx(-1.0)


def test_clv_se_calcula_con_closing_odds(ledger):
    # Conseguimos 2.10 y cerró en 2.00 → CLV = 2.10/2.00 - 1 = +5%.
    pid = _record(ledger, odds=2.10)
    ledger.settle(pid, won=True, closing_odds=2.00)
    s = ledger.summary()
    assert s.clv_avg == pytest.approx(0.05)


def test_settle_pick_inexistente_falla(ledger):
    with pytest.raises(KeyError):
        ledger.settle(999, won=True)


def test_summary_vacio(ledger):
    s = ledger.summary()
    assert s.total == 0
    assert s.roi == 0.0
    assert s.clv_avg is None
