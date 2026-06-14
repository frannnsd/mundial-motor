"""Bot conversable de Telegram: escribile un partido y te responde la predicción.

Uso:  python scripts/run_chat_bot.py   (queda escuchando; Ctrl+C para cortar)

Ejemplos de lo que le podés escribir:
  • "Argentina vs México"
  • "Brasil - Croacia"
  • /hoy   (predicciones de todos los partidos de hoy)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.brain import HELP, build_today_message, load_brain  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402


async def main() -> None:
    settings = get_settings()
    if not settings.has_telegram:
        print("Falta TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID en .env.")
        sys.exit(1)

    print("Cargando el cerebro (modelos)...")
    brain = load_brain()

    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode
    from aiogram.filters import Command
    from aiogram.types import Message

    bot = Bot(
        settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    @dp.message(Command("start", "help", "ayuda"))
    async def _start(message: Message) -> None:
        await message.answer(HELP)

    @dp.message(Command("hoy"))
    async def _hoy(message: Message) -> None:
        date_str = datetime.now().strftime("%d/%m/%Y")
        await message.answer(build_today_message(brain, settings, date_str=date_str))

    @dp.message()
    async def _any(message: Message) -> None:
        await message.answer(brain.handle_text(message.text or ""))

    print("✅ Bot escuchando. Escribile por Telegram (ej: 'Argentina vs México').")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
