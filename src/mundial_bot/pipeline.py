"""Orquestador del bot — conecta los 6 agentes en un flujo end-to-end.

datos (Elo+Dixon-Coles) → cuotas (The Odds API) → de-vig → detección de value →
staking ¼ Kelly → combinadas → cartilla Telegram → ledger.

Los modelos y los partidos se pueden inyectar (para testear offline). En operación
real, los partidos salen del feed de The Odds API (que ya trae fixtures + cuotas).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from mundial_bot.collectors.results import load_results
from mundial_bot.config import Settings, get_settings
from mundial_bot.ledger.store import Ledger
from mundial_bot.models.elo_model import EloModel
from mundial_bot.models.goals_model import GoalsModel
from mundial_bot.notify.formatting import format_daily_card
from mundial_bot.staking.kelly import StakeConfig, StakedPick, size_portfolio
from mundial_bot.staking.parlays import (
    Parlay,
    highest_payout_parlay,
    safest_parlay,
    suggest_parlays,
)
from mundial_bot.value.devig import devig
from mundial_bot.value.ev import Selection, ValuePick, evaluate
from mundial_bot.value.odds import MatchOdds, OddsClient, best_1x2, best_book_for
from mundial_bot.value.team_aliases import normalize_team

logger = logging.getLogger(__name__)

# Los partidos del Mundial se juegan en cancha neutral.
WORLD_CUP_NEUTRAL = True
# Nombre del resultado de empate en The Odds API.
_DRAW_NAME = "Draw"
# Stake de combinadas: fijo y chico (varianza alta).
PARLAY_STAKE_PCT = 0.01


@dataclass
class Models:
    elo: EloModel
    goals: GoalsModel | None = None


@dataclass
class PipelineResult:
    n_matches: int
    value_picks: list[ValuePick]
    staked: list[StakedPick]
    parlays: list[tuple[str, str, Parlay, float]]
    message: str


def build_models(*, since: str = "1994-01-01", fit_goals: bool = True) -> Models:
    """Entrena Elo (1X2) y, opcionalmente, Dixon-Coles (mercados de goles)."""
    df = load_results(since=since)
    elo = EloModel().fit(df)
    goals = None
    if fit_goals:
        recent = df[df["date"] >= pd.Timestamp("2018-01-01")].reset_index(drop=True)
        try:
            goals = GoalsModel().fit(recent)
        except Exception as exc:  # noqa: BLE001 — fallback a Elo, pero logueamos
            logger.warning("GoalsModel.fit falló; sigo solo con Elo: %s", exc)
            goals = None
    return Models(elo=elo, goals=goals)


def value_picks_for_match(
    match: MatchOdds, models: Models, *, min_edge: float
) -> list[ValuePick]:
    """Detecta los value picks 1X2 de un partido (Elo vs cuotas de-vigueadas)."""
    odds_1x2 = best_1x2(match)
    if set(odds_1x2) != {"home", "draw", "away"}:
        return []  # mercado incompleto

    home = normalize_team(match.home_team)
    away = normalize_team(match.away_team)
    p = models.elo.predict(home, away, neutral=WORLD_CUP_NEUTRAL)
    model_probs = {"home": p.home, "draw": p.draw, "away": p.away}
    fair = devig(odds_1x2, method="shin")

    api_name = {"home": match.home_team, "away": match.away_team, "draw": _DRAW_NAME}
    picks: list[ValuePick] = []
    for outcome in ("home", "draw", "away"):
        sel = Selection(
            match=match.match,
            market="1X2",
            selection=outcome,
            odds=odds_1x2[outcome],
            bookmaker=best_book_for(match, api_name[outcome]) or "?",
        )
        vp = evaluate(sel, model_probs[outcome], fair_prob=fair[outcome])
        if vp.edge >= min_edge:
            picks.append(vp)
    return picks


def _build_parlays(
    value_picks: list[ValuePick], *, bankroll: float, min_edge: float
) -> list[tuple[str, str, Parlay, float]]:
    """Arma la combinada conservadora y la de alto riesgo a partir de los value picks."""
    suggestions = suggest_parlays(value_picks, sizes=(2, 3), min_combined_ev=min_edge)
    if not suggestions:
        return []
    stake = round(PARLAY_STAKE_PCT * bankroll, 2)
    safe = safest_parlay(suggestions)
    risky = highest_payout_parlay(suggestions)
    out: list[tuple[str, str, Parlay, float]] = []
    if safe:
        out.append(("Conservadora", "🔒", safe, stake))
    if risky and risky is not safe:
        out.append(("Alto riesgo", "🚀", risky, stake))
    return out


def run_pipeline(
    *,
    settings: Settings | None = None,
    models: Models | None = None,
    matches: list[MatchOdds] | None = None,
    date_str: str | None = None,
    record_ledger: bool = False,
) -> PipelineResult:
    """Corre el pipeline completo y devuelve los picks + la cartilla lista para enviar."""
    settings = settings or get_settings()
    models = models or build_models()

    if matches is None:
        if not settings.has_odds_api:
            raise RuntimeError("Sin ODDS_API_KEY: pasá `matches=` o cargá la clave en .env.")
        matches = OddsClient(settings.odds_api_key, settings.odds_region).get_matches()

    value_picks: list[ValuePick] = []
    for m in matches:
        try:
            value_picks.extend(value_picks_for_match(m, models, min_edge=settings.min_edge))
        except Exception as exc:  # noqa: BLE001 — equipo sin rating, mercado raro, etc.
            logger.debug("Salteo partido %s: %s", getattr(m, "match", "?"), exc)
            continue
    value_picks.sort(key=lambda p: p.edge, reverse=True)

    cfg = StakeConfig(
        bankroll=settings.bankroll_usd,
        kelly_fraction=settings.kelly_fraction,
        max_stake_pct=settings.max_stake_pct,
        max_total_exposure_pct=settings.max_total_exposure_pct,
    )
    staked = size_portfolio(value_picks, cfg)
    parlays = _build_parlays(
        value_picks, bankroll=settings.bankroll_usd, min_edge=settings.min_edge
    )

    date_str = date_str or datetime.now().strftime("%d/%m/%Y")
    message = format_daily_card(
        staked, parlays, bankroll=settings.bankroll_usd, date_str=date_str
    )

    if record_ledger:
        _record_to_ledger(staked, parlays, date_str=date_str)

    return PipelineResult(
        n_matches=len(matches),
        value_picks=value_picks,
        staked=staked,
        parlays=parlays,
        message=message,
    )


def _record_to_ledger(
    staked: list[StakedPick],
    parlays: list[tuple[str, str, Parlay, float]],
    *,
    date_str: str,
) -> None:
    """Persiste los picks sugeridos en el ledger (status pending)."""
    with Ledger() as lg:
        for s in staked:
            sel = s.pick.selection
            lg.record(
                created_at=date_str, match=sel.match, market=sel.market,
                selection=sel.selection, odds=sel.odds, bookmaker=sel.bookmaker,
                model_prob=s.pick.model_prob, fair_prob=s.pick.fair_prob,
                edge=s.pick.edge, stake=s.stake, kind="single",
            )
        for label, _emoji, par, stake in parlays:
            lg.record(
                created_at=date_str, match=f"{label} ({par.n_legs} patas)", market="PARLAY",
                selection="+".join(leg.selection.selection for leg in par.legs),
                odds=par.combined_odds, model_prob=par.combined_prob,
                edge=par.combined_ev, stake=stake, kind="parlay",
            )
