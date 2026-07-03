"""Props MLB (M3): Ks del abridor + hits/HR de bateadores — coherentes y point-in-time.

Tres props, de más a menos confiable:

1. Ks DEL ABRIDOR (el prop de béisbol): μ = tasa de Ks por out decaída
   (halflife 200 días) × outs esperados. Los outs esperados son la media
   decaída de innings del pitcher, con shrinkage hacia 5.0 IP (k = 3 starts):
   con pocos starts manda la duración típica de un abridor, no su muestra.
   Dispersión: NegBin con el Fano de SUS Ks por start (floor 1.0), pasada por
   `research.distributions.count_pmf` — la MISMA vara que el resto del repo.

2. HITS DE BATEADORES — reparto COHERENTE: el total de hits del EQUIPO (viene
   del cerebro unificado M1, acá se recibe como parámetro) se reparte entre
   los 9 del lineup proporcional a (tasa de hits por PA decaída y shrunk ×
   PAs esperados por slot). PROPIEDAD: Σ μ_bateadores == μ_equipo EXACTO.
   La dispersión individual es Poisson(μ) — misma aproximación documentada
   que players/props.py de fútbol.

3. P(HR): 1 − e^{−μ_HR} con μ_HR = tasa HR/PA decaída, shrunk hacia la media
   de liga (0.032/PA, k = 100 PAs) × PAs del slot. ES EL PROP MÁS RUIDOSO:
   el base rate es bajo (~3% por PA), así que hasta con media temporada de
   muestra el shrinkage pesa mucho — usarlo como señal débil, no como pick.

Todo point-in-time: `as_of` es obligatorio y SOLO cuentan partidos con fecha
estrictamente anterior. Los gamelogs pueden inyectarse (tests sin red) o se
buscan cache-primero vía collectors.mlb_players.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mundial_bot.collectors import mlb_players
from mundial_bot.research.distributions import count_pmf

HALFLIFE_DAYS = 200.0          # decay temporal de tasas por jugador
KS_GRID_MAX = 20               # grilla de la pmf de Ks (cubre >99.9% de la masa)
LEAGUE_K_PER_OUT = 8.2 / 27.0  # K/9 de liga ~8.2 → Ks por out (fallback sin datos)
DEFAULT_OUTS = 15.0            # 5.0 innings: duración típica de un start moderno
OUTS_SHRINK_K = 3.0            # starts-equivalentes de fe en los 15 outs
LEAGUE_HIT_RATE_PA = 0.230     # hits por PA de liga (prior del shrinkage)
LEAGUE_HR_RATE_PA = 0.032      # HR por PA de liga
RATE_SHRINK_PA_K = 100.0       # PAs-equivalentes de fe en la media de liga

# PAs esperados por slot del orden de bateo 1..9: los primeros batean más veces
# por partido (~4.7 el leadoff vs ~3.9 el noveno). Pesos fijos documentados,
# consistentes con los PA/juego históricos por slot de MLB.
PA_BY_SLOT = (4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9)


def _past_games(gamelog: list[dict], as_of: pd.Timestamp) -> list[dict]:
    """Solo partidos ESTRICTAMENTE anteriores a as_of (point-in-time)."""
    return [g for g in gamelog if g.get("date") and pd.Timestamp(g["date"]) < as_of]


def _decay_weight(date, as_of: pd.Timestamp, halflife: float = HALFLIFE_DAYS) -> float:
    days = max((as_of - pd.Timestamp(date)).days, 0)
    return float(0.5 ** (days / halflife))


def _fetch_multi_season(fetch, person_id: int, seasons: tuple[int, ...]) -> list[dict]:
    out: list[dict] = []
    for season in seasons:
        out.extend(fetch(person_id, season))
    return out


def _default_seasons(as_of: pd.Timestamp) -> tuple[int, int]:
    return (as_of.year, as_of.year - 1)


# ---------------------------------------------------------------------------
# 1) Ks del abridor
# ---------------------------------------------------------------------------


def pitcher_ks_distribution(
    person_id: int,
    *,
    as_of: pd.Timestamp | str,
    seasons: tuple[int, ...] | None = None,
    gamelog: list[dict] | None = None,
) -> tuple[float, float, np.ndarray]:
    """(μ, var, pmf) de los Ks del abridor en su próximo start, point-in-time.

    μ = (Ks por out decaído, halflife 200d) × (outs esperados: media decaída de
    outs por start, shrunk hacia 15 con k=3 starts). var = μ × Fano de sus Ks
    por start (floor 1.0) → NegBin/Poisson vía count_pmf (la vara del repo).

    `gamelog` inyectable para tests sin red; si es None se buscan las
    `seasons` (default: año de as_of y el previo) cache-primero.
    Sin starts previos: tasa de liga (~0.30 K/out) × 15 outs, Fano 1.0.
    """
    as_of = pd.Timestamp(as_of)
    if gamelog is None:
        gamelog = _fetch_multi_season(
            mlb_players.fetch_pitcher_gamelog, person_id, seasons or _default_seasons(as_of)
        )
    starts = [g for g in _past_games(gamelog, as_of) if g.get("is_start")]

    if not starts:
        mu = LEAGUE_K_PER_OUT * DEFAULT_OUTS
        var = mu  # Fano 1.0 → Poisson
        return mu, var, count_pmf(mu, var, KS_GRID_MAX)

    w_sum = k_sum = out_sum = 0.0
    for g in starts:
        w = _decay_weight(g["date"], as_of)
        w_sum += w
        k_sum += w * float(g["strikeouts"])
        out_sum += w * float(g["outs"])

    k_per_out = (k_sum / out_sum) if out_sum > 0 else LEAGUE_K_PER_OUT
    raw_outs = out_sum / w_sum if w_sum > 0 else DEFAULT_OUTS
    exp_outs = (w_sum * raw_outs + OUTS_SHRINK_K * DEFAULT_OUTS) / (w_sum + OUTS_SHRINK_K)
    mu = max(k_per_out * exp_outs, 1e-6)

    ks = np.array([float(g["strikeouts"]) for g in starts])
    fano = 1.0
    if len(ks) >= 2 and ks.mean() > 0:
        fano = max(float(ks.var() / ks.mean()), 1.0)
    var = mu * fano
    return mu, var, count_pmf(mu, var, KS_GRID_MAX)


# ---------------------------------------------------------------------------
# 2) Hits de bateadores — reparto coherente del total del equipo
# ---------------------------------------------------------------------------


def batter_hit_rate_pa(
    gamelog: list[dict],
    *,
    as_of: pd.Timestamp | str,
    league_rate: float = LEAGUE_HIT_RATE_PA,
    shrink_k: float = RATE_SHRINK_PA_K,
) -> float:
    """Tasa de hits por PA decaída (halflife 200d) con shrinkage hacia la liga.

        rate = (PA_eff × rate_cruda + k × liga) / (PA_eff + k),  k = 100 PAs

    Con ~10 PAs el resultado queda pegado a la liga; con ~400 PAs domina la
    tasa propia del bateador. Point-in-time: solo partidos con fecha < as_of.
    """
    as_of = pd.Timestamp(as_of)
    num = den = 0.0
    for g in _past_games(gamelog, as_of):
        w = _decay_weight(g["date"], as_of)
        num += w * float(g["hits"])
        den += w * float(g["plate_appearances"])
    raw = num / den if den > 0 else league_rate
    return (den * raw + shrink_k * league_rate) / (den + shrink_k)


def batter_hits_props(
    team_hits_mean: float,
    lineup: list[dict],
    batter_rates: dict[int, list[dict]],
    *,
    as_of: pd.Timestamp | str,
) -> pd.DataFrame:
    """Reparte el total de hits del EQUIPO entre los 9 del lineup (coherente).

    `team_hits_mean` viene del cerebro unificado M1 (acá no se recalcula).
    `lineup` = filas de collectors.mlb_players.fetch_game_lineup (person_id,
    full_name, batting_order 1..9). `batter_rates` = gamelogs de bateo por
    person_id (de fetch_batter_gamelog, o sintéticos en tests); un bateador
    sin gamelog usa la tasa de liga.

    Peso de cada bateador = tasa de hits/PA (decaída + shrunk) × PAs esperados
    de su slot (PA_BY_SLOT). μ_i = total × peso_i / Σ pesos, así que
    Σ μ_i == team_hits_mean EXACTO (propiedad de coherencia, tol 1e-9).
    P(1+ hit) = 1 − pmf[0] con dispersión Poisson(μ) — aproximación documentada.
    """
    as_of = pd.Timestamp(as_of)
    if not lineup:
        return pd.DataFrame()

    rows: list[dict] = []
    for entry in lineup:
        slot = int(entry["batting_order"])
        if not 1 <= slot <= 9:
            raise ValueError(f"batting_order fuera de 1..9: {slot}")
        rate = batter_hit_rate_pa(
            batter_rates.get(int(entry["person_id"]), []), as_of=as_of
        )
        exp_pa = PA_BY_SLOT[slot - 1]
        rows.append({
            "person_id": int(entry["person_id"]),
            "name": entry.get("full_name", ""),
            "batting_order": slot,
            "exp_pa": exp_pa,
            "hit_rate_pa": rate,
            "weight": rate * exp_pa,
        })

    df = pd.DataFrame(rows)
    denom = float(df["weight"].sum())
    df["mu_hits"] = team_hits_mean * df["weight"] / denom if denom > 0 else 0.0
    df["p_hit_1plus"] = 1.0 - (-df["mu_hits"]).map(math.exp)  # 1 − pmf[0] Poisson
    return (
        df.drop(columns=["weight"])
        .sort_values("batting_order")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3) P(HR) — el prop más ruidoso
# ---------------------------------------------------------------------------


def batter_hr_prob(
    person_id: int,
    *,
    as_of: pd.Timestamp | str,
    batting_order: int = 5,
    seasons: tuple[int, ...] | None = None,
    gamelog: list[dict] | None = None,
) -> float:
    """P(al menos 1 HR) = 1 − e^{−μ_HR}, point-in-time.

    μ_HR = tasa HR/PA decaída (halflife 200d) shrunk hacia la liga
    (0.032/PA, k = 100 PAs) × PAs esperados del slot. ADVERTENCIA: es el prop
    MÁS RUIDOSO de la capa — el base rate es tan bajo que la tasa individual
    casi nunca se separa de la liga con significancia; tratarlo como señal
    débil, nunca como pick por sí solo.
    """
    as_of = pd.Timestamp(as_of)
    if not 1 <= int(batting_order) <= 9:
        raise ValueError(f"batting_order fuera de 1..9: {batting_order}")
    if gamelog is None:
        gamelog = _fetch_multi_season(
            mlb_players.fetch_batter_gamelog, person_id, seasons or _default_seasons(as_of)
        )
    num = den = 0.0
    for g in _past_games(gamelog, as_of):
        w = _decay_weight(g["date"], as_of)
        num += w * float(g["home_runs"])
        den += w * float(g["plate_appearances"])
    raw = num / den if den > 0 else LEAGUE_HR_RATE_PA
    rate = (den * raw + RATE_SHRINK_PA_K * LEAGUE_HR_RATE_PA) / (den + RATE_SHRINK_PA_K)
    mu_hr = rate * PA_BY_SLOT[int(batting_order) - 1]
    return 1.0 - math.exp(-mu_hr)
