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


def scan_today(settings: Settings, brain: BotBrain) -> str:
    """Escaneo del día: analiza TODOS los mercados de todos los partidos y devuelve las
    mejores jugadas (más firmes / modelo>casa / batacazos) + combinadas.

    Sin value gatekeeping: no descarta nada por falta de edge, solo ordena qué mostrar.
    Junta las casas de API-Football con las de odds-api.io (si hay key).
    """
    from mundial_bot.collectors.odds_af import fetch_odds, merge_odds
    from mundial_bot.evaluator import best_plays, build_combos, format_full_scan, plays_from_book
    from mundial_bot.models.market_book import build_market_book

    key = settings.api_football_key
    # Partidos POR JUGAR de los próximos días (los terminados/en vivo tienen cuotas
    # viejas o de in-play que no sirven para apostar pre-partido). Acotado para no
    # disparar demasiadas consultas ni mensajes gigantes.
    try:
        window = get_schedule(settings, days_back=0, days_ahead=2)
        fixtures = sorted((f for f in window if f.upcoming), key=lambda f: f.date)[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("No pude traer la agenda; uso fixtures de hoy: %s", exc)
        fixtures = [f for f in fetch_today_fixtures(settings) if getattr(f, "upcoming", True)]

    # Cuotas extra de odds-api.io (una sola bajada de eventos del Mundial, reutilizada).
    extra_events = None
    if settings.has_oddspapi:
        try:
            from mundial_bot.collectors.odds_oddspapi import fetch_wc_events

            extra_events = fetch_wc_events(settings.oddspapi_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("odds-api.io sin eventos: %s", exc)

    all_plays = []
    for f in fixtures:
        if not f.fixture_id:
            continue
        book = build_market_book(
            normalize_team(f.home_team), normalize_team(f.away_team),
            elo=brain.models.elo, goals=brain.models.goals,
            corners=brain.corners, cards=brain.cards,
            referee=f.referee, knockout=f.knockout, neutral=True, match_name=f.match,
        )
        odds: dict = {}
        try:
            odds = fetch_odds(key, f.fixture_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin cuotas para %s: %s", f.match, exc)
        if extra_events is not None:
            try:
                from mundial_bot.collectors.odds_oddspapi import fetch_match_odds

                extra = fetch_match_odds(
                    settings.oddspapi_key, f.home_team, f.away_team, events=extra_events
                )
                if extra:
                    odds = merge_odds(odds, extra) if odds else extra
            except Exception as exc:  # noqa: BLE001
                logger.warning("odds-api.io sin cuotas para %s: %s", f.match, exc)
        all_plays.extend(plays_from_book(book, odds))

    firmes, mejor, batacazos = best_plays(all_plays)
    likely, payout = build_combos(firmes + mejor)
    return format_full_scan(
        firmes, mejor, batacazos, likely, payout,
        date_str=datetime.now(UTC).strftime("%d/%m/%Y"),
    )


def find_fixture_id(settings: Settings, home: str, away: str) -> int | None:
    """Resuelve el fixture_id de un partido buscando por nombres en una ventana de fechas."""
    want = frozenset({normalize_team(home), normalize_team(away)})
    try:
        fixtures = get_schedule(settings, days_back=1, days_ahead=7)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No pude resolver fixture para %s vs %s: %s", home, away, exc)
        return None
    return next(
        (f.fixture_id for f in fixtures
         if frozenset({normalize_team(f.home_team), normalize_team(f.away_team)}) == want
         and f.fixture_id),
        None,
    )


def injuries_for_match(settings: Settings, home: str, away: str) -> str:
    """Bajas (lesionados/suspendidos) de un partido, por equipo. Texto para el agente."""
    if not settings.has_api_football:
        return "(Sin API-Football para traer las bajas.)"
    from mundial_bot.collectors.injuries import fetch_injuries

    fixture_id = find_fixture_id(settings, home, away)
    if fixture_id is None:
        return "(No encontré el fixture para traer las bajas.)"
    try:
        injuries = fetch_injuries(settings.api_football_key, fixture_id=fixture_id)
    except Exception as exc:  # noqa: BLE001
        return f"(No pude traer las bajas: {exc})"
    if not injuries:
        return "Sin bajas reportadas para este partido (o la API todavía no las cargó)."
    lines = ["BAJAS (lesionados/suspendidos) — pesá vos qué jugador es importante:"]
    for team, players in injuries.items():
        names = ", ".join(f"{p.player} ({p.reason})" for p in players[:12])
        lines.append(f"  {team}: {names}")
    return "\n".join(lines)


def odds_for_match(settings: Settings, home: str, away: str) -> dict:
    """Cuotas REALES de un partido (API-Football por fixture + odds-api.io), fusionadas.

    Resuelve el fixture buscando el partido en una ventana de fechas por nombre de equipo.
    Devuelve {mercado: MarketOdds} en el formato de API-Football (o {} si no hay).
    """
    from mundial_bot.collectors.odds_af import fetch_odds, merge_odds

    key = settings.api_football_key
    odds: dict = {}
    fixture_id = find_fixture_id(settings, home, away)
    if fixture_id and key:
        try:
            odds = fetch_odds(key, fixture_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin cuotas API-Football para %s vs %s: %s", home, away, exc)
    if settings.has_oddspapi:
        try:
            from mundial_bot.collectors.odds_oddspapi import fetch_match_odds

            extra = fetch_match_odds(settings.oddspapi_key, home, away)
            if extra:
                odds = merge_odds(odds, extra) if odds else extra
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sin cuotas odds-api.io para %s vs %s: %s", home, away, exc)
    return odds


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
