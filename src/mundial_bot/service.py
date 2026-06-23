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
    try:
        snapshot_clv(settings, brain)   # guarda la cuota de apertura de los picks (CLV)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CLV snapshot falló: %s", exc)
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
    from mundial_bot.evaluator import best_plays, build_combos, format_full_scan

    all_plays = _day_plays(settings, brain)
    firmes, mejor, batacazos = best_plays(all_plays)
    likely, payout = build_combos(firmes + mejor)
    return format_full_scan(
        firmes, mejor, batacazos, likely, payout,
        date_str=datetime.now(UTC).strftime("%d/%m/%Y"),
    )


def _day_plays(settings: Settings, brain: BotBrain) -> list:
    """Junta TODAS las jugadas priced de los partidos por jugar (próximos 2 días)."""
    from mundial_bot.collectors.odds_af import fetch_odds, merge_odds
    from mundial_bot.evaluator import plays_from_book
    from mundial_bot.models.market_book import build_market_book

    key = settings.api_football_key
    try:
        window = get_schedule(settings, days_back=0, days_ahead=2)
        fixtures = sorted((f for f in window if f.upcoming), key=lambda f: f.date)[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("No pude traer la agenda; uso fixtures de hoy: %s", exc)
        fixtures = [f for f in fetch_today_fixtures(settings) if getattr(f, "upcoming", True)]

    extra_events = None
    if settings.has_oddspapi:
        try:
            from mundial_bot.collectors.odds_oddspapi import fetch_wc_events

            extra_events = fetch_wc_events(settings.oddspapi_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("odds-api.io sin eventos: %s", exc)

    all_plays: list = []
    for f in fixtures:
        if not f.fixture_id:
            continue
        book = build_market_book(
            normalize_team(f.home_team), normalize_team(f.away_team),
            elo=brain.models.elo, goals=brain.models.goals,
            corners=brain.corners, cards=brain.cards, shots=brain.shots,
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
    return all_plays


def day_parlays(settings: Settings, brain: BotBrain) -> str:
    """Arma VARIAS combinadas mezclando los partidos del día (cross-match)."""
    from mundial_bot.evaluator import best_plays, build_combos, format_day_parlays

    firmes, mejor, _ = best_plays(_day_plays(settings, brain))
    likely, payout = build_combos(
        firmes + mejor, sizes=(2, 3, 4, 5), top_likely=6, top_payout=6
    )
    return format_day_parlays(
        likely, payout, date_str=datetime.now(UTC).strftime("%d/%m/%Y")
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


def match_scorers(settings: Settings, brain: BotBrain, home: str, away: str) -> str:
    """Goleadores probables de un partido: P(1+/2+/3+ goles) por jugador de cada equipo.

    Reparte el xG del equipo (Dixon-Coles, ya ajustado por el rival) entre los jugadores
    según su tasa de gol de la temporada.
    """
    if not settings.has_api_football:
        return "(Sin API-Football para traer los planteles.)"
    if brain.models.goals is None:
        return "(No tengo el modelo de goles cargado.)"
    from mundial_bot.collectors.player_stats import (
        fetch_squad_goals,
        format_scorers,
        goalscorer_probs,
        team_id_map,
    )

    rh, ra = brain.resolve(home), brain.resolve(away)
    if rh not in brain.known or ra not in brain.known:
        return f"(No tengo a {home} o {away} en el modelo del Mundial.)"
    if not brain.models.goals.can_predict(rh, ra):
        return f"(No tengo datos de goles para {rh} o {ra}.)"
    try:
        _, home_xg, away_xg = brain.models.goals.score_matrix(rh, ra, neutral=True)
    except Exception as exc:  # noqa: BLE001
        return f"(No pude calcular el xG: {exc})"

    key = settings.api_football_key
    try:
        idmap = team_id_map(key)
    except Exception as exc:  # noqa: BLE001
        return f"(No pude traer los planteles: {exc})"

    blocks = []
    for team, xg in ((rh, home_xg), (ra, away_xg)):
        tid = idmap.get(team)
        if tid is None:
            blocks.append(f"⚽ {team}: no encontré el plantel.")
            continue
        try:
            squad = fetch_squad_goals(key, tid)
        except Exception:  # noqa: BLE001
            blocks.append(f"⚽ {team}: no pude traer el plantel.")
            continue
        blocks.append(format_scorers(team, goalscorer_probs(squad, xg)))
    head = f"🥅 GOLEADORES — {rh} vs {ra}\n(asume titulares; tasa de la temporada)\n"
    return head + "\n\n".join(blocks)


def snapshot_clv(settings: Settings, brain: BotBrain, *, max_per_fixture: int = 6) -> int:
    """Guarda la cuota de APERTURA de los picks firmes de los próximos partidos (para CLV)."""
    if not settings.has_api_football:
        return 0
    from mundial_bot.clv import ClvStore
    from mundial_bot.collectors.odds_af import fetch_odds
    from mundial_bot.evaluator import best_plays, plays_from_book
    from mundial_bot.models.market_book import build_market_book

    key = settings.api_football_key
    try:
        window = get_schedule(settings, days_back=0, days_ahead=2)
        fixtures = sorted((f for f in window if f.upcoming), key=lambda f: f.date)[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("CLV snapshot sin fixtures: %s", exc)
        return 0

    now = datetime.now(UTC).isoformat()
    logged = 0
    with ClvStore() as store:
        for f in fixtures:
            if not f.fixture_id:
                continue
            try:
                odds = fetch_odds(key, f.fixture_id)
            except Exception:  # noqa: BLE001
                continue
            if not odds:
                continue
            book = build_market_book(
                normalize_team(f.home_team), normalize_team(f.away_team),
                elo=brain.models.elo, goals=brain.models.goals,
                corners=brain.corners, cards=brain.cards, shots=brain.shots,
                referee=f.referee, knockout=f.knockout, neutral=True, match_name=f.match,
            )
            firmes, mejor, _ = best_plays(plays_from_book(book, odds))
            seen: set[tuple[str, str]] = set()
            for p in (firmes + mejor):
                if not p.odds_key or (p.market, p.pick) in seen or len(seen) >= max_per_fixture:
                    continue
                seen.add((p.market, p.pick))
                logged += store.open_pick(
                    opened_at=now, fixture_id=f.fixture_id, match=f.match,
                    market=p.odds_key[0], outcome=p.odds_key[1], pick=p.pick,
                    open_odds=p.odd, open_book=p.book,
                )
    return logged


def close_clv(settings: Settings, *, window_min: float = 120.0) -> int:
    """Captura la cuota de CIERRE de los picks cuyo partido arranca pronto (para CLV)."""
    if not settings.has_api_football:
        return 0
    from mundial_bot.clv import ClvStore
    from mundial_bot.collectors.odds_af import fetch_odds

    key = settings.api_football_key
    try:
        window = get_schedule(settings, days_back=0, days_ahead=2)
    except Exception:  # noqa: BLE001
        return 0
    now = datetime.now(UTC).isoformat()
    closed = 0
    with ClvStore() as store:
        for f in window:
            if not f.fixture_id or not f.upcoming:
                continue
            mins = _minutes_to_kickoff(f.date)
            if mins is None or not (0 < mins <= window_min):
                continue
            rows = store.open_for_fixture(f.fixture_id)
            if not rows:
                continue
            try:
                odds = fetch_odds(key, f.fixture_id)
            except Exception:  # noqa: BLE001
                continue
            for r in rows:
                mo = odds.get(r["market"])
                if mo and r["outcome"] in mo.best:
                    close_odds, close_book = mo.best[r["outcome"]]
                    store.set_close(
                        r["id"], closed_at=now, close_odds=close_odds,
                        close_book=close_book, open_odds=r["open_odds"],
                    )
                    closed += 1
    return closed


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
    try:
        close_clv(settings)   # captura la cuota de cierre de los picks cerca del inicio (CLV)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CLV close falló: %s", exc)
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
                corners=brain.corners, cards=brain.cards, shots=brain.shots,
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
