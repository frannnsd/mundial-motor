"""Tests de props MLB (M3) — SIN red: gamelogs sintéticos, cero llamadas HTTP."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.collectors.mlb_players import ip_to_outs
from mundial_bot.players.mlb_props import (
    LEAGUE_HIT_RATE_PA,
    PA_BY_SLOT,
    batter_hit_rate_pa,
    batter_hits_props,
    batter_hr_prob,
    pitcher_ks_distribution,
)
from mundial_bot.research.distributions import p_over

AS_OF = pd.Timestamp("2026-07-02")

# ---------------------------------------------------------------------------
# Helpers sintéticos
# ---------------------------------------------------------------------------


def _start(date: str, ks: int, outs: int = 18, *, is_start: bool = True) -> dict:
    """Un start sintético de pitcheo (outs=18 → 6.0 IP)."""
    return {
        "date": date, "game_pk": 0, "is_start": is_start, "strikeouts": ks,
        "outs": outs, "innings": outs / 3.0, "batters_faced": outs + 6,
        "hits": 5, "home_runs": 1, "earned_runs": 2,
    }


def _bat_game(date: str, hits: int, pa: int, hr: int = 0) -> dict:
    """Un partido sintético de bateo."""
    return {
        "date": date, "game_pk": 0, "hits": hits, "home_runs": hr,
        "at_bats": pa, "plate_appearances": pa,
    }


def _lineup_9() -> list[dict]:
    return [
        {"person_id": 100 + i, "full_name": f"bat{i}", "batting_order": i}
        for i in range(1, 10)
    ]


# ---------------------------------------------------------------------------
# 1) Conversión de innings: la notación .1/.2 son tercios, no décimas
# ---------------------------------------------------------------------------


def test_ip_a_outs_conversion():
    assert ip_to_outs("5.2") == 17  # 5⅔ entradas
    assert ip_to_outs("6.0") == 18
    assert ip_to_outs("0.1") == 1
    assert ip_to_outs("7") == 21
    assert ip_to_outs(None) == 0


# ---------------------------------------------------------------------------
# 2) Point-in-time: starts con fecha >= as_of NO cuentan
# ---------------------------------------------------------------------------


def test_pitcher_point_in_time_excluye_futuro():
    past = [_start(f"2026-06-{d:02d}", 5) for d in (5, 10, 15, 20, 25)]
    future = [_start("2026-07-02", 15), _start("2026-07-07", 15)]  # >= as_of

    mu_solo_pasado, _, _ = pitcher_ks_distribution(1, as_of=AS_OF, gamelog=past)
    mu_con_futuro, _, _ = pitcher_ks_distribution(1, as_of=AS_OF, gamelog=past + future)

    # Si el futuro se filtrara mal, los 15 Ks inflarían mu (fallaría fuerte).
    assert mu_con_futuro == pytest.approx(mu_solo_pasado, abs=1e-12)


def test_batter_point_in_time_excluye_futuro():
    past = [_bat_game(f"2026-06-{d:02d}", 2, 5) for d in range(1, 21)]
    future = [_bat_game("2026-07-02", 5, 5)]
    rate_pasado = batter_hit_rate_pa(past, as_of=AS_OF)
    rate_con_futuro = batter_hit_rate_pa(past + future, as_of=AS_OF)
    assert rate_con_futuro == pytest.approx(rate_pasado, abs=1e-12)


# ---------------------------------------------------------------------------
# 3) COHERENCIA: la suma de los 9 μ == total del equipo, exacto
# ---------------------------------------------------------------------------


def test_coherencia_suma_hits_lineup_igual_total_equipo():
    rates = {
        100 + i: [_bat_game(f"2026-06-{d:02d}", (i % 3), 4) for d in range(1, 25)]
        for i in range(1, 10)
    }
    team_mean = 8.5
    props = batter_hits_props(team_mean, _lineup_9(), rates, as_of=AS_OF)

    assert len(props) == 9
    assert abs(float(props["mu_hits"].sum()) - team_mean) < 1e-9
    # P(1+ hit) coherente con Poisson: en (0, 1) y creciente con mu.
    assert ((props["p_hit_1plus"] > 0) & (props["p_hit_1plus"] < 1)).all()
    orden = props.sort_values("mu_hits")["p_hit_1plus"].to_numpy()
    assert (orden[1:] >= orden[:-1]).all()


def test_coherencia_con_bateadores_sin_gamelog():
    # Sin datos, todos caen a la tasa de liga — la coherencia se mantiene igual.
    props = batter_hits_props(8.5, _lineup_9(), {}, as_of=AS_OF)
    assert abs(float(props["mu_hits"].sum()) - 8.5) < 1e-9
    assert props["hit_rate_pa"].to_numpy() == pytest.approx(LEAGUE_HIT_RATE_PA)
    # El leadoff (más PAs) recibe más mu que el noveno.
    mu = props.set_index("batting_order")["mu_hits"]
    assert mu[1] > mu[9]


# ---------------------------------------------------------------------------
# 4) Ks: ordinalidad K/9 y pmf válida
# ---------------------------------------------------------------------------


def test_ks_pitcher_dominante_da_mu_mayor_y_pmf_suma_1():
    fechas = [f"2026-06-{d:02d}" for d in (1, 6, 11, 16, 21, 26)]
    alto = [_start(f, 10) for f in fechas]  # 10 K / 6 IP → K/9 = 15
    bajo = [_start(f, 3) for f in fechas]   # 3 K / 6 IP → K/9 = 4.5

    mu_alto, var_alto, pmf_alto = pitcher_ks_distribution(1, as_of=AS_OF, gamelog=alto)
    mu_bajo, var_bajo, pmf_bajo = pitcher_ks_distribution(2, as_of=AS_OF, gamelog=bajo)

    assert mu_alto > mu_bajo
    assert abs(float(pmf_alto.sum()) - 1.0) < 1e-9
    assert abs(float(pmf_bajo.sum()) - 1.0) < 1e-9
    assert var_alto >= mu_alto  # Fano floor 1.0 → nunca sub-Poisson
    # El dominante pasa la línea 5.5 mucho más seguido.
    assert p_over(pmf_alto, 5.5) > p_over(pmf_bajo, 5.5) + 0.3


def test_ks_sin_starts_usa_liga():
    mu, var, pmf = pitcher_ks_distribution(1, as_of=AS_OF, gamelog=[])
    assert 3.0 < mu < 6.5  # ~0.30 K/out × 15 outs ≈ 4.6
    assert var == pytest.approx(mu)
    # Los relevos (is_start=False) tampoco cuentan como starts.
    relevos = [_start("2026-06-10", 3, 6, is_start=False)]
    mu_rel, _, _ = pitcher_ks_distribution(1, as_of=AS_OF, gamelog=relevos)
    assert mu_rel == pytest.approx(mu)


# ---------------------------------------------------------------------------
# 5) Shrinkage: poca muestra → liga; mucha muestra → tasa propia
# ---------------------------------------------------------------------------


def test_shrinkage_bateador_10_pa_cerca_de_liga_400_pa_cerca_de_su_tasa():
    propia = 0.5  # tasa cruda irrealmente alta para separar bien
    poco = [_bat_game("2026-06-30", 3, 5), _bat_game("2026-07-01", 2, 5)]  # 10 PA
    mucho = [_bat_game(f"2026-{m:02d}-{d:02d}", 2, 4)
             for m in (4, 5, 6) for d in range(1, 29)]  # 84 juegos < as_of
    mucho += [_bat_game(f"2026-03-{d:02d}", 2, 4) for d in range(12, 28)]  # 100 j, 400 PA

    r_poco = batter_hit_rate_pa(poco, as_of=AS_OF)
    r_mucho = batter_hit_rate_pa(mucho, as_of=AS_OF)

    assert abs(r_poco - LEAGUE_HIT_RATE_PA) < abs(r_poco - propia)
    assert r_poco < 0.27  # pegado a la liga (0.23)
    assert abs(r_mucho - propia) < abs(r_mucho - LEAGUE_HIT_RATE_PA)
    assert r_mucho > 0.42  # cerca de su 0.5


def test_hr_prob_shrinkage_y_rango():
    # Slugger con 400 PA y 0.08 HR/PA vs bateador sin datos (liga pura).
    slugger = [_bat_game(f"2026-{m:02d}-{d:02d}", 1, 4, hr=(1 if d % 3 == 0 else 0))
               for m in (4, 5, 6) for d in range(1, 29)]
    p_slugger = batter_hr_prob(1, as_of=AS_OF, batting_order=3, gamelog=slugger)
    p_liga = batter_hr_prob(2, as_of=AS_OF, batting_order=3, gamelog=[])

    assert 0.0 < p_liga < p_slugger < 0.5  # ruidoso pero ordenado y acotado
    # Sin datos: exactamente 1 - exp(-0.032 × PA del slot 3).
    import math
    assert p_liga == pytest.approx(1.0 - math.exp(-0.032 * PA_BY_SLOT[2]))


def test_batting_order_invalido_falla_rapido():
    with pytest.raises(ValueError):
        batter_hr_prob(1, as_of=AS_OF, batting_order=0, gamelog=[])
    lineup = [{"person_id": 1, "full_name": "x", "batting_order": 10}]
    with pytest.raises(ValueError):
        batter_hits_props(8.5, lineup, {}, as_of=AS_OF)
