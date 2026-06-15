"""Handlers del bot conversable, compartidos por el bot local y el servicio de deploy.

`BrainHolder` permite recargar el cerebro (cuando el ciclo diario lo actualiza) sin
reiniciar el bot: los handlers siempre leen `holder.brain`.
"""

from __future__ import annotations

import asyncio
import html
from dataclasses import dataclass, field
from datetime import date, datetime

from mundial_bot.brain import HELP, BotBrain, build_today_message
from mundial_bot.config import Settings


@dataclass
class BrainHolder:
    brain: BotBrain
    history: list = field(default_factory=list)  # memoria de la charla con el agente


def register_handlers(dp, settings: Settings, holder: BrainHolder) -> None:
    """Registra todos los handlers en el Dispatcher de aiogram."""
    from aiogram.filters import Command
    from aiogram.types import Message

    @dp.message(Command("start", "help", "ayuda"))
    async def _start(message: Message) -> None:
        await message.answer(HELP)

    @dp.message(Command("hoy"))
    async def _hoy(message: Message) -> None:
        date_str = datetime.now().strftime("%d/%m/%Y")
        await message.answer(build_today_message(holder.brain, settings, date_str=date_str))

    @dp.message(Command("balance"))
    async def _balance(message: Message) -> None:
        from mundial_bot.tracking import PredictionStore, format_balance, grade_pending

        if settings.has_api_football:
            grade_pending(settings.api_football_key)
        with PredictionStore() as store:
            await message.answer(format_balance(store.balance()))

    @dp.message(Command("apuesta"))
    async def _apuesta(message: Message) -> None:
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
        text = message.text or ""
        if settings.has_anthropic:
            from mundial_bot.agent import ask_agent

            reply = await asyncio.to_thread(
                ask_agent, text, settings=settings, brain=holder.brain, history=holder.history
            )
            # El agente responde en texto plano; lo escapamos para que el modo HTML no rompa.
            await message.answer(html.escape(reply))
        else:
            await message.answer(holder.brain.handle_text(text))
