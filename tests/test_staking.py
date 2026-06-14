"""Tests del gestor de riesgo: Kelly + combinadas (Agente 4)."""

from __future__ import annotations

import pytest

from mundial_bot.staking.kelly import (
    StakeConfig,
    kelly_fraction,
    size_portfolio,
    stake_for,
)
from mundial_bot.staking.parlays import (
    suggest_parlays,
    highest_payout_parlay,
    safest_parlay,
)
from mundial_bot.value.ev import Selection, ValuePick


def _pick(match: str, prob: float, odds: float) -> ValuePick:
    sel = Selection(match=match, market="1X2", selection="home", odds=odds)
    return ValuePick(selection=sel, model_prob=prob, edge=prob * odds - 1.0)


# ---------- Kelly ----------

def test_kelly_fraction_formula():
    # p=0.55, o=2.0 → f* = (1.1-1)/(2-1) = 0.10
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)


def test_kelly_fraction_cero_sin_edge():
    assert kelly_fraction(0.45, 2.0) == 0.0   # edge negativo → no apostar


def test_stake_for_aplica_cuarto_kelly_y_tope():
    cfg = StakeConfig(bankroll=100, kelly_fraction=0.25, max_stake_pct=0.03)
    # full kelly 0.10 → ¼ = 0.025 → 2.5% de 100 = $2.50 (bajo el tope 3%)
    assert stake_for(_pick("A vs B", 0.55, 2.0), cfg) == pytest.approx(2.50)


def test_stake_for_respeta_tope_por_apuesta():
    cfg = StakeConfig(bankroll=100, kelly_fraction=0.25, max_stake_pct=0.03)
    # Edge enorme → ¼ Kelly excede 3% → se topa en $3.00
    big = _pick("A vs B", 0.90, 3.0)   # full kelly alto
    assert stake_for(big, cfg) == pytest.approx(3.00)


def test_size_portfolio_reescala_por_exposicion_total():
    cfg = StakeConfig(bankroll=100, kelly_fraction=1.0, max_stake_pct=1.0,
                      max_total_exposure_pct=0.25)
    picks = [_pick(f"M{i} vs X", 0.60, 2.0) for i in range(5)]  # cada uno full kelly 0.20

    staked = size_portfolio(picks, cfg)
    total = sum(s.stake for s in staked)

    # Suma sin tope = 5 * 0.20 = 1.0 (100%) → reescala a 25% = $25
    assert total == pytest.approx(25.0, abs=0.05)


# ---------- Combinadas ----------

def test_combinada_de_patas_value_es_ev_positivo():
    # Dos patas +EV independientes → EV combinado positivo.
    picks = [_pick("A vs B", 0.55, 2.0), _pick("C vs D", 0.55, 2.0)]
    parlays = suggest_parlays(picks, sizes=(2,))

    assert len(parlays) == 1
    par = parlays[0]
    assert par.combined_odds == pytest.approx(4.0)
    assert par.combined_prob == pytest.approx(0.3025)
    assert par.combined_ev > 0   # 0.3025*4 - 1 = 0.21


def test_suggest_parlays_no_combina_mismo_partido():
    # Dos selecciones del MISMO partido no deben combinarse (correlacionadas).
    a = ValuePick(Selection("A vs B", "1X2", "home", 2.0), 0.55, 0.10)
    b = ValuePick(Selection("A vs B", "1X2", "draw", 3.5), 0.30, 0.05)
    parlays = suggest_parlays([a, b], sizes=(2,))
    assert parlays == []


def test_safest_y_highest_payout():
    picks = [
        _pick("A vs B", 0.70, 1.6),   # alta prob, baja cuota
        _pick("C vs D", 0.40, 3.0),   # baja prob, alta cuota
        _pick("E vs F", 0.55, 2.0),
    ]
    parlays = suggest_parlays(picks, sizes=(2, 3))
    assert parlays  # hay combinadas +EV

    safe = safest_parlay(parlays)
    risky = highest_payout_parlay(parlays)
    # La más segura tiene mayor prob combinada que la de mayor pago.
    assert safe.combined_prob >= risky.combined_prob
    # La de alto riesgo tiene mayor cuota combinada.
    assert risky.combined_odds >= safe.combined_odds
