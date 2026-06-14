"""Ciclos programados (diario + pre-partido), reutilizables por el servicio y los scripts.

Centraliza la lógica que corre sola en producción:
  - `daily_cycle`: refresca datos, califica lo de ayer, recarga el cerebro
    (autoalimentado) y arma el balance + las predicciones del día.
  - `prematch_alerts`: ~1-2h antes de cada partido manda la predicción fresca + las bajas.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

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
