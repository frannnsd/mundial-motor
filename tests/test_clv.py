"""Tests del tracking de CLV (closing line value)."""

from __future__ import annotations

import pytest

from mundial_bot.clv import ClvStore, format_clv


@pytest.fixture()
def store():
    s = ClvStore(":memory:")
    yield s
    s.close()


def test_open_es_idempotente(store):
    kw = dict(opened_at="t0", fixture_id=1, match="A vs B", market="Match Winner",
              outcome="Home", pick="Gana A", open_odds=2.0, open_book="X")
    assert store.open_pick(**kw) == 1
    assert store.open_pick(**kw) == 0          # no duplica la apertura


def test_clv_positivo_cuando_el_cierre_baja(store):
    store.open_pick(opened_at="t0", fixture_id=1, match="A vs B", market="Match Winner",
                    outcome="Home", pick="Gana A", open_odds=2.20, open_book="X")
    row = store.open_for_fixture(1)[0]
    # Cerró a 2.00 (la cuota se acortó) → le ganamos al cierre.
    store.set_close(row["id"], closed_at="t1", close_odds=2.00, close_book="Y",
                    open_odds=2.20)
    s = store.summary()
    assert s.n_closed == 1 and s.positive == 1
    assert s.avg_clv == pytest.approx(2.20 / 2.00 - 1)   # +10%


def test_clv_negativo_cuando_el_cierre_sube(store):
    store.open_pick(opened_at="t0", fixture_id=2, match="C vs D", market="Match Winner",
                    outcome="Away", pick="Gana D", open_odds=1.80, open_book="X")
    row = store.open_for_fixture(2)[0]
    store.set_close(row["id"], closed_at="t1", close_odds=2.00, close_book="Y",
                    open_odds=1.80)
    s = store.summary()
    assert s.positive == 0 and s.avg_clv < 0


def test_format_clv_sin_cierres(store):
    assert "todavía sin cierres" in format_clv(store.summary())
