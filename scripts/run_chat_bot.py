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

    @dp.message(Command("balance"))
    async def _balance(message: Message) -> None:
        from mundial_bot.tracking import PredictionStore, format_balance, grade_pending

        if settings.has_api_football:
            grade_pending(settings.api_football_key)
        with PredictionStore() as store:
            await message.answer(format_balance(store.balance()))

    @dp.message(Command("apuesta"))
    async def _apuesta(message: Message) -> None:
        from datetime import date

        from mundial_bot.betlog import BetStore, parse_bet_command

        try:
            stake, odds, desc = parse_bet_command(message.text or "")
        except ValueError as exc:
            await message.answer(f"❌ {exc}\nEjemplo: <code>/apuesta 5 2.10 Argentina gana</code>")
            return
        with BetStore() as store:
            bet_id = store.log(
                created_at=date.today().isoformat(), description=desc, stake=stake, odds=odds
            )
        await message.answer(
            f"✅ Anotada #{bet_id}: {desc} · ${stake:.2f} @ {odds:.2f}\n"
            f"Cuando se defina: /gane {bet_id} o /perdi {bet_id}"
        )

    async def _settle_bet(message: Message, *, won: bool) -> None:
        from mundial_bot.betlog import BetStore

        parts = (message.text or "").split()
        if len(parts) < 2 or not parts[1].isdigit():
            await message.answer("Decime el número. Ej: <code>/gane 3</code>")
            return
        with BetStore() as store:
            try:
                store.settle(int(parts[1]), won=won)
            except KeyError:
                await message.answer(f"No existe la apuesta #{parts[1]}.")
                return
        await message.answer(f"✅ #{parts[1]} marcada como {'GANADA 🟢' if won else 'perdida 🔴'}.")

    @dp.message(Command("gane", "gano"))
    async def _gane(message: Message) -> None:
        await _settle_bet(message, won=True)

    @dp.message(Command("perdi", "perdio"))
    async def _perdi(message: Message) -> None:
        await _settle_bet(message, won=False)

    @dp.message(Command("roi", "apuestas"))
    async def _roi(message: Message) -> None:
        from mundial_bot.betlog import BetStore, format_roi

        with BetStore() as store:
            opens = store.open_bets()
            msg = format_roi(store.summary())
        if opens:
            msg += "\n\n<b>Abiertas:</b>\n" + "\n".join(
                f"#{b['id']}: {b['description']} (${b['stake']:.0f} @ {b['odds']:.2f})"
                for b in opens
            )
        await message.answer(msg)

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
