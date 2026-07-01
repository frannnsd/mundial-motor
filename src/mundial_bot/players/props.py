"""Props por jugador: minutos esperados + reparto coherente del total del equipo.

La capa NO recalcula totales de equipo: recibe `team_totals` (del cerebro
unificado de la Fase A o de cualquier fuente {cantidad: (media, var)}) y los
REPARTE entre jugadores:

    μ_jugador = share_normalizado × total_equipo × (min_esperados / horizonte)

donde el share se normaliza sobre el "equipo-partido esperado" — es decir,
tal que Σ_jugadores μ_jugador == media_equipo EXACTO (propiedad de COHERENCIA,
tolerancia 1e-6). La varianza del total del equipo no se propaga por jugador:
la dispersión individual se modela Poisson(μ) — aproximación documentada.

SIN LEAKAGE DE XI: `lineup_confirmed` (ids del XI publicado ~20-40 min antes
del kickoff) SOLO puede pasarse para predicciones posteriores a su publicación.
Con `None` se usa el XI probable: los minutos recientes de cada jugador
(promedio sobre todas sus citaciones, que ya descuenta rotación y suplencias).

Probabilidades derivadas (independencia aproximada entre conteos, documentado):
  P(anota) = 1 − e^{−μ_goles}
  P(anota o asiste) = 1 − e^{−(μ_goles + μ_asistencias)}  (cota superior leve:
      ignora la correlación negativa gol/asistencia en la misma jugada)
  P(tarjeta) = 1 − e^{−(μ_amarillas + μ_rojas)}  (rojas solo si vienen en totals)
  P(2+ remates) = 1 − e^{−μ}(1 + μ)
"""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_SUB_MINUTES = 15.0  # suplente sin historial de minutos como suplente
_HORIZONS = {"90": 90.0, "120": 120.0}


def expected_minutes(
    player_row, *, horizon: str = "90", lineup_confirmed: set[int] | None = None
) -> float:
    """Minutos esperados de un jugador para el próximo partido.

    - Sin XI confirmado (`lineup_confirmed=None`): XI probable → promedio de
      minutos recientes sobre TODAS sus citaciones (`min_avg`), que ya pondera
      titularidades, suplencias y partidos sin entrar.
    - Confirmado EN el XI: base 90 ajustada por su patrón de sustitución
      histórico — simple: min(90, media de minutos como titular); 90 si nunca
      arrancó (sin historial no se le inventa una sustitución).
    - Confirmado AFUERA del XI: media de minutos como suplente, o 15 si nunca
      entró desde el banco.

    `horizon="120"` (eliminación directa con alargue) escala por 120/90.
    """
    base_h = _HORIZONS.get(horizon)
    if base_h is None:
        raise ValueError(f"horizon inválido: {horizon!r} (usar '90' o '120')")

    if lineup_confirmed is None:
        minutes = float(player_row["min_avg"])
    elif int(player_row["player_id"]) in lineup_confirmed:
        starter_avg = float(player_row["min_avg_starter"])
        minutes = min(90.0, starter_avg) if starter_avg > 0 else 90.0
    else:
        sub_avg = float(player_row["min_avg_sub"])
        minutes = sub_avg if sub_avg > 0 else DEFAULT_SUB_MINUTES

    return max(0.0, min(minutes, 90.0)) * (base_h / 90.0)


def match_props(
    team_totals: dict[str, tuple[float, float]],
    shares_df: pd.DataFrame,
    *,
    horizon: str = "90",
    lineup_confirmed: set[int] | None = None,
) -> pd.DataFrame:
    """Reparte los totales del equipo entre sus jugadores (medias + probs Poisson).

    `team_totals` = {stat: (media, var)} del partido para ESTE equipo, en el
    MISMO horizonte que `horizon`. La var no se usa por jugador (Poisson por μ,
    documentado arriba). `shares_df` es la salida de `shares.player_shares`.

    COHERENCIA: para cada stat, Σ_jugadores μ == media del equipo (exacto salvo
    flotante), con o sin `lineup_confirmed` — el share efectivo se normaliza
    sobre el equipo-partido esperado: μ_i = media × (rate_i·min_i) / Σ_j(rate_j·min_j).
    """
    if shares_df.empty:
        return pd.DataFrame()
    horizon_min = _HORIZONS.get(horizon)
    if horizon_min is None:
        raise ValueError(f"horizon inválido: {horizon!r} (usar '90' o '120')")
    unknown = [s for s in team_totals if f"rate_{s}" not in shares_df.columns]
    if unknown:
        raise ValueError(f"stats sin rate en shares_df: {unknown}")

    out = shares_df[["player_id", "player_name", "position"]].copy()
    out["exp_minutes"] = [
        expected_minutes(row, horizon=horizon, lineup_confirmed=lineup_confirmed)
        for _, row in shares_df.iterrows()
    ]

    for stat, (mean, _var) in team_totals.items():
        weights = shares_df[f"rate_{stat}"].to_numpy() * out["exp_minutes"].to_numpy()
        denom = float(weights.sum())
        out[f"mu_{stat}"] = mean * weights / denom if denom > 0 else 0.0

    _add_derived_probs(out, set(team_totals))
    sort_by = "mu_shots" if "shots" in team_totals else "exp_minutes"
    return out.sort_values(sort_by, ascending=False).reset_index(drop=True)


def _add_derived_probs(props: pd.DataFrame, stats: set[str]) -> None:
    """Agrega las probabilidades Poisson derivadas para los stats disponibles."""
    if "goals" in stats:
        props["p_scores"] = 1.0 - (-props["mu_goals"]).map(math.exp)
    if "goals" in stats and "assists" in stats:
        mu = props["mu_goals"] + props["mu_assists"]
        props["p_goal_or_assist"] = 1.0 - (-mu).map(math.exp)
    if "yellow" in stats:
        mu = props["mu_yellow"] + (props["mu_red"] if "red" in stats else 0.0)
        props["p_card"] = 1.0 - (-mu).map(math.exp)
    if "shots" in stats:
        mu = props["mu_shots"]
        props["p_shots_2plus"] = 1.0 - (1.0 + mu) * (-mu).map(math.exp)
    if "sot" in stats:
        props["p_sot_1plus"] = 1.0 - (-props["mu_sot"]).map(math.exp)
