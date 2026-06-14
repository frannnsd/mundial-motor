"""Servicio 24/7 para deploy: bot conversable + scheduler (diario + pre-partido).

Un solo proceso = una sola base SQLite compartida (predicciones, apuestas, alertas).
Entrypoint del Dockerfile.

Uso:  python scripts/run_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.brain import load_brain  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.handlers import BrainHolder, register_handlers  # noqa: E402
from mundial_bot.notify.telegram_bot import _split_message  # noqa: E402
from mundial_bot.service import daily_cycle, prematch_alerts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")


async def main() -> None:
    settings = get_settings()
    if not settings.has_telegram:
        print("Falta TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID en .env.")
        sys.exit(1)

    logger.info("Cargando el cerebro...")
    holder = BrainHolder(load_brain())

    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    bot = Bot(
        settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    register_handlers(dp, settings, holder)

    async def _send(text: str) -> None:
        for chunk in _split_message(text):
            await bot.send_message(settings.telegram_chat_id, chunk)

    async def daily_job() -> None:
        try:
            message, brain = await asyncio.to_thread(daily_cycle, settings)
            holder.brain = brain  # cerebro recargado/autoalimentado
            await _send(message)
            logger.info("Ciclo diario OK.")
        except Exception:
            logger.exception("Fallo en el ciclo diario")

    async def prematch_job() -> None:
        try:
            messages = await asyncio.to_thread(prematch_alerts, settings, holder.brain)
            for msg in messages:
                await _send(msg)
            if messages:
                logger.info("Pre-partido: %d alertas enviadas.", len(messages))
        except Exception:
            logger.exception("Fallo en el ciclo pre-partido")

    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    scheduler.add_job(
        daily_job,
        CronTrigger(hour=settings.daily_picks_hour, minute=0, timezone=settings.timezone),
        misfire_grace_time=3600,
    )
    scheduler.add_job(prematch_job, IntervalTrigger(minutes=30))
    scheduler.start()

    logger.info("✅ Servicio activo: bot escuchando + scheduler corriendo.")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
