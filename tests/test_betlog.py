"""Tests del registro de apuestas (ROI real)."""

from __future__ import annotations

import pytest

from mundial_bot.betlog import BetStore, format_roi, parse_bet_command


def test_log_settle_y_roi():
    store = BetStore(":memory:")
    try:
        b1 = store.log(created_at="2026-06-14", description="Argentina gana", stake=10, odds=2.0)
        store.settle(b1, won=True)
        b2 = store.log(created_at="2026-06-14", description="Over 2.5", stake=10, odds=1.8)
        store.settle(b2, won=False)

        s = store.summary()
        assert s.settled == 2
        assert s.won == 1
        assert s.staked == 20
        assert s.returned == 20          # 10*2.0 ganada; la perdida cobra 0
        assert s.roi == pytest.approx(0.0)
        assert "ROI" in format_roi(s)
    finally:
        store.close()


def test_parse_bet_command():
    stake, odds, desc = parse_bet_command("/apuesta 5 2.10 Argentina gana")
    assert stake == 5.0
    assert odds == 2.10
    assert desc == "Argentina gana"


def test_parse_bet_command_invalido():
    with pytest.raises(ValueError):
        parse_bet_command("/apuesta 5")


def test_log_rechaza_valores_invalidos():
    store = BetStore(":memory:")
    try:
        with pytest.raises(ValueError):
            store.log(created_at="x", description="y", stake=0, odds=2.0)
        with pytest.raises(ValueError):
            store.log(created_at="x", description="y", stake=5, odds=0.9)
    finally:
        store.close()
