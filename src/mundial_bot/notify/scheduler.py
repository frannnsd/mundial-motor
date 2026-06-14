"""Scheduler diario con APScheduler — Agente 5.

La Bot API de Telegram no tiene envío programado nativo: lo corremos nosotros.
Un CronTrigger dispara el job a una hora local fija (zona horaria explícita, porque
los partidos del Mundial cruzan husos de USA/Canadá/México).
"""

from __future__ import annotations

from collections.abc import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def start_daily_scheduler(
    job: Callable[[], None], *, hour: int, timezone: str, minute: int = 0
) -> None:
    """Arranca un scheduler bloqueante que corre `job` todos los días a `hour:minute`."""
    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(
        job,
        CronTrigger(hour=hour, minute=minute, timezone=timezone),
        name="cartilla-diaria",
        misfire_grace_time=3600,  # tolera 1h de retraso (ej. tras un reinicio)
    )
    print(f"⏰ Scheduler activo: cartilla diaria a las {hour:02d}:{minute:02d} ({timezone}).")
    scheduler.start()
