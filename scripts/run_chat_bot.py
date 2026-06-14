"""Bot conversable de Telegram (uso local / pruebas). Para 24/7 usá run_bot.py.

Escribile un partido y te responde. Comandos: /hoy /balance /apuesta /roi.

Uso:  python scripts/run_chat_bot.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.brain import load_brain  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.handlers import BrainHolder, register_handlers  # noqa: E402


async def main() -> None:
    settings = get_settings()
    if not settings.has_telegram:
        print("Falta TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID en .env.")
        sys.exit(1)

    print("Cargando el cerebro (modelos)...")
    holder = BrainHolder(load_brain())

    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode

    bot = Bot(
        settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    register_handlers(dp, settings, holder)

    print("✅ Bot escuchando. Escribile por Telegram (ej: 'Argentina vs México').")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
