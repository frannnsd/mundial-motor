"""Scheduler de los jobs del Mundial (APScheduler, todo en UTC).

Cadencia (UTC / hora argentina):
  daily    12:00 (09:00 AR)         payload web + predicciones del día
  lineups  cada 5 min               barato: sin partidos en ventana, no llama a nada
  settle   06:00 (03:00 AR)         liquida los partidos de AYER en AR — un día AR
                                    abarca DOS fechas UTC (la noche AR cae en el día
                                    UTC siguiente), así que se liquidan ambas;
                                    run_settle es idempotente.
  weekly   domingo 13:00 (10:00 AR) resumen del forward-test + backup

MLB (mismo scheduler, jobs de wc/mlb_jobs.py):
  mlb_daily  15:00 (12:00 AR)  los juegos MLB arrancan ~17-23 UTC
  mlb_settle 09:00 (06:00 AR)  los juegos terminan de madrugada AR; run_mlb_settle
                               sin fecha liquida AYER en UTC (el schedule-date real)

Catch-up honesto al arrancar (el free tier de Render se reinicia): si ya pasaron
las 12:00 UTC y hoy no hay daily_reports, se corre run_daily una vez (ídem
mlb_daily con su hora y sus reports sport='mlb').

El orquestador decide si arranca: `start_if_enabled()` mira WC_SCHEDULER == "1".
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from mundial_bot.wc import jobs, mlb_jobs, store

logger = logging.getLogger(__name__)

AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
DAILY_HOUR_UTC = 12
SETTLE_HOUR_UTC = 6
WEEKLY_HOUR_UTC = 13
LINEUPS_EVERY_MIN = 5
MLB_DAILY_HOUR_UTC = 15
MLB_SETTLE_HOUR_UTC = 9
MISFIRE_GRACE_S = 3600  # tras un reinicio, un cron atrasado <1h igual corre

_scheduler: BackgroundScheduler | None = None


def _settle_previous_day_ar() -> None:
    """Liquida el día de AYER en AR cubriendo las dos fechas UTC que ese día toca."""
    yesterday_ar = (datetime.now(AR_TZ) - timedelta(days=1)).date()
    for d in (yesterday_ar, yesterday_ar + timedelta(days=1)):
        jobs.run_settle(d.isoformat())


def _needs_daily_catchup() -> bool:
    """¿Ya pasaron las 12:00 UTC de hoy sin daily_reports guardados?"""
    return _needs_catchup(DAILY_HOUR_UTC, "wc")


def _needs_mlb_daily_catchup() -> bool:
    """¿Ya pasaron las 15:00 UTC de hoy sin daily_reports sport='mlb'?"""
    return _needs_catchup(MLB_DAILY_HOUR_UTC, "mlb")


def _needs_catchup(hour_utc: int, sport: str) -> bool:
    if not store.is_configured():
        return False
    now = datetime.now(UTC)
    if now.hour < hour_utc:
        return False
    try:
        return not store.get_reports(now.strftime("%Y-%m-%d"), sport=sport)
    except requests.RequestException as exc:
        logger.warning("Catch-up %s: no pude consultar daily_reports (%s); lo salteo.",
                       sport, exc)
        return False


def start_scheduler() -> BackgroundScheduler:
    """Arranca (una sola vez) el BackgroundScheduler con los 4 jobs del Mundial."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    sched = BackgroundScheduler(timezone="UTC")
    common = {"misfire_grace_time": MISFIRE_GRACE_S, "coalesce": True, "max_instances": 1}
    sched.add_job(jobs.run_daily, "cron", hour=DAILY_HOUR_UTC, minute=0,
                  id="wc_daily", **common)
    sched.add_job(jobs.run_lineups, "interval", minutes=LINEUPS_EVERY_MIN,
                  id="wc_lineups", **common)
    sched.add_job(_settle_previous_day_ar, "cron", hour=SETTLE_HOUR_UTC, minute=0,
                  id="wc_settle", **common)
    sched.add_job(jobs.run_weekly, "cron", day_of_week="sun", hour=WEEKLY_HOUR_UTC,
                  minute=0, id="wc_weekly", **common)
    sched.add_job(mlb_jobs.run_mlb_daily, "cron", hour=MLB_DAILY_HOUR_UTC, minute=0,
                  id="mlb_daily", **common)
    # sin fecha, run_mlb_settle liquida AYER en UTC (el schedule-date que terminó).
    sched.add_job(mlb_jobs.run_mlb_settle, "cron", hour=MLB_SETTLE_HOUR_UTC, minute=0,
                  id="mlb_settle", **common)
    if _needs_daily_catchup():
        logger.info("Catch-up: hoy no hay daily_reports y ya pasaron las 12 UTC; "
                    "corro run_daily una vez.")
        sched.add_job(jobs.run_daily, id="wc_daily_catchup", **common)
    if _needs_mlb_daily_catchup():
        logger.info("Catch-up MLB: hoy no hay reports sport='mlb' y ya pasaron "
                    "las 15 UTC; corro run_mlb_daily una vez.")
        sched.add_job(mlb_jobs.run_mlb_daily, id="mlb_daily_catchup", **common)

    sched.start()
    _scheduler = sched
    logger.info("Scheduler WC arrancado: daily 12:00, lineups cada %d min, "
                "settle 06:00, weekly dom 13:00, mlb_daily 15:00, "
                "mlb_settle 09:00 (todo UTC).", LINEUPS_EVERY_MIN)
    return sched


def start_if_enabled() -> BackgroundScheduler | None:
    """Arranca el scheduler solo si WC_SCHEDULER == "1" (decisión del orquestador)."""
    if os.environ.get("WC_SCHEDULER") != "1":
        logger.info("WC_SCHEDULER != '1': el scheduler del Mundial queda apagado.")
        return None
    return start_scheduler()


def scheduler_status() -> dict:
    """Últimos runs registrados + próximos fire times de cada job."""
    sched = _scheduler
    running = bool(sched is not None and sched.running)
    next_runs = []
    if running:
        for j in sched.get_jobs():
            next_runs.append({
                "id": j.id,
                "next_run_utc": j.next_run_time.isoformat() if j.next_run_time else None,
            })
    last_runs: list[dict] = []
    if store.is_configured():
        try:
            last_runs = store.latest_job_runs()
        except requests.RequestException as exc:
            logger.warning("scheduler_status: no pude leer job_runs: %s", exc)
    return {"running": running, "jobs": next_runs, "last_runs": last_runs}
