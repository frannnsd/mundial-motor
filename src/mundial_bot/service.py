"""Ciclos programados (diario + pre-partido), reutilizables por el servicio y los scripts.

Centraliza la lógica que corre sola en producción:
  - `daily_cycle`: refresca datos, califica lo de ayer, recarga el cerebro
    (autoalimentado) y arma el balance + las predicciones del día.
  - `prematch_alerts`: ~1-2h antes de cada partido manda la predicción fresca + las bajas.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from mundial_bot.brain import BotBrain, build_today_message, fetch_today_fixtures, load_brain
from mundial_bot.collectors.injuries import fetch_injuries
from mundial_bot.collectors.team_stats import build_cache as refresh_team_stats
from mundial_bot.collectors.wc_results import build_cache as refresh_wc_results
from mundial_bot.config import Settings
from mundial_bot.report import build_match_report, format_match_report
from mundial_bot.tracking import PredictionStore, format_balance, grade_pending
from mundial_bot.value.team_aliases import normalize_team

logger = logging.getLogger(__name__)

PREMATCH_WINDOW_FROM = 20
PREMATCH_WINDOW_TO = 150


def daily_cycle(settings: Settings) -> tuple[str, BotBrain]:
    """Refresca datos + califica + recarga cerebro. Devuelve (mensaje, cerebro nuevo)."""
    key = settings.api_football_key
    if key:
        try:
            refresh_team_stats(key, last=12)
            refresh_wc_results(key)
            grade_pending(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallo refrescando datos: %s", exc)

    brain = load_brain()
    with PredictionStore() as store:
        balance = format_balance(store.balance())
    today = build_today_message(brain, settings, date_str=datetime.now(UTC).strftime("%d/%m/%Y"))
    return balance + "\n\n" + today, brain


def evaluate_today(settings: Settings, brain: BotBrain, *, min_ev: float = 0.0) -> str:
    """Evalúa todos los partidos de hoy contra el mercado real → cuotas buenas + combinadas.

    Junta las casas de API-Football con las de odds-api.io (si hay key) y se queda con la
    cuota más alta de cada resultado: lee desde la primera hasta la última cuota.
    """
    from mundial_bot.collectors.odds_af import fetch_odds, merge_odds
    from mundial_bot.evaluator import build_parlays, evaluate_match, format_evaluation

    key = settings.api_football_key
    fixtures = fetch_today_fixtures(settings)

    # Cuotas extra de odds-api.io (una sola bajada de eventos del Mundial, reutilizada).
    extra_events = None
    if settings.has_oddspapi:
        try:
            from mundial_bot.collectors.odds_oddspapi import fetch_wc_events

            extra_events = fetch_wc_events(settings.oddspapi_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("odds-api.io sin eventos: %s", exc)

    all_bets = []
    for f in fixtures:
        if not f.fixture_id:
            continue
        report = build_match_report(
            normalize_team(f.home_team), normalize_team(f.away_team),
            elo=brain.models.elo, goals=brain.models.goals,
            corners=brain.corners, cards=brain.cards,
            referee=f.referee, knockout=f.knockout, neutral=True, match_name=f.match,
        )
        try:
            odds = fetch_odds(key, f.fixture_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin cuotas para %s: %s", f.match, exc)
            continue
        if extra_events is not None:
            try:
                from mundial_bot.collectors.odds_oddspapi import fetch_match_odds

                extra = fetch_match_odds(
                    settings.oddspapi_key, f.home_team, f.away_team, events=extra_events
                )
                if extra:
                    odds = merge_odds(odds, extra)
            except Exception as exc:  # noqa: BLE001
                logger.warning("odds-api.io sin cuotas para %s: %s", f.match, exc)
        all_bets.extend(evaluate_match(report, odds, min_ev=min_ev))

    parlays = build_parlays(all_bets, min_ev=min_ev)
    return format_evaluation(all_bets, parlays, date_str=datetime.now(UTC).strftime("%d/%m/%Y"))


def get_schedule(settings: Settings, *, days_back: int = 1, days_ahead: int = 4):
    """Trae los partidos del Mundial en una ventana (jugados + en vivo + por jugar)."""
    from mundial_bot.collectors.fixtures import FixturesClient

    today = datetime.now(UTC).date()
    return FixturesClient(settings.api_football_key).get_range(
        date_from=(today - timedelta(days=days_back)).isoformat(),
        date_to=(today + timedelta(days=days_ahead)).isoformat(),
    )


def format_schedule(fixtures, *, tz_name: str, date_str: str) -> str:
    """Agenda: en vivo / por jugar (con horario local) / jugados (con resultado)."""
    tz = ZoneInfo(tz_name)

    def local_time(f) -> str:
        try:
            return pd.to_datetime(f.date, utc=True).tz_convert(tz).strftime("%d/%m %H:%M")
        except Exception:  # noqa: BLE001
            return "?"

    live = [f for f in fixtures if f.live]
    upcoming = sorted((f for f in fixtures if f.upcoming), key=lambda f: f.date)
    played = sorted((f for f in fixtures if f.played), key=lambda f: f.date)

    lines = [f"📅 <b>AGENDA MUNDIAL — {date_str}</b>"]
    if live:
        lines.append("\n🔴 <b>EN VIVO</b>")
        lines += [f"   {f.match} ({f.home_goals}-{f.away_goals})" for f in live]
    if upcoming:
        lines.append("\n⏳ <b>POR JUGAR</b>")
        lines += [f"   {local_time(f)} · {f.match}" for f in upcoming]
    if played:
        lines.append("\n✅ <b>JUGADOS</b>")
        lines += [f"   {f.match} <b>{f.home_goals}-{f.away_goals}</b>" for f in played[-12:]]
    return "\n".join(lines)


def _minutes_to_kickoff(date_str: str) -> float | None:
    try:
        kickoff = pd.to_datetime(date_str, utc=True)
        return (kickoff - pd.Timestamp.now(tz="UTC")).total_seconds() / 60.0
    except Exception:  # noqa: BLE001
        return None


def _injuries_text(key: str, fixture_id: int) -> str:
    try:
        injuries = fetch_injuries(key, fixture_id=fixture_id)
    except Exception:  # noqa: BLE001
        return ""
    if not injuries:
        return ""
    lines = ["🚑 <b>Bajas (decidí vos el impacto):</b>"]
    for team, players in injuries.items():
        names = ", ".join(p.player for p in players[:6])
        lines.append(f"   {team}: {names}")
    return "\n".join(lines)


def prematch_alerts(settings: Settings, brain: BotBrain) -> list[str]:
    """Arma las alertas pre-partido pendientes (partidos por empezar, no avisados aún)."""
    key = settings.api_football_key
    messages: list[str] = []
    fixtures = fetch_today_fixtures(settings)
    with PredictionStore() as store:
        for f in fixtures:
            if not f.fixture_id or store.was_alerted(f.fixture_id):
                continue
            mins = _minutes_to_kickoff(f.date)
            if mins is None or not (PREMATCH_WINDOW_FROM <= mins <= PREMATCH_WINDOW_TO):
                continue
            report = build_match_report(
                normalize_team(f.home_team), normalize_team(f.away_team),
                elo=brain.models.elo, goals=brain.models.goals,
                corners=brain.corners, cards=brain.cards,
                referee=f.referee, knockout=f.knockout, neutral=True, match_name=f.match,
            )
            msg = f"⏰ <b>PRE-PARTIDO</b> (empieza en ~{int(mins)} min)\n\n"
            msg += format_match_report(report)
            injuries = _injuries_text(key, f.fixture_id)
            if injuries:
                msg += "\n\n" + injuries
            messages.append(msg)
            store.mark_alerted(f.fixture_id, datetime.now(UTC).isoformat())
    return messages
