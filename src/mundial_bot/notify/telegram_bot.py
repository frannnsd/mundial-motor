"""Envío de mensajes por Telegram con aiogram — Agente 5.

El envío real necesita TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID (van en .env).
La función es async porque aiogram lo es; el pipeline la corre con asyncio.run.

Modo dry-run: si no hay token, imprime el mensaje por consola en vez de enviarlo,
así se puede probar el flujo completo sin credenciales.
"""

from __future__ import annotations

import asyncio

# Límite de Telegram por mensaje (caracteres). Partimos si excede.
TELEGRAM_MAX_CHARS = 4096


def _split_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Parte un mensaje largo en trozos <= limit, respetando saltos de línea."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            chunks.append(current.rstrip("\n"))
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current.rstrip("\n"))
    return chunks


async def send_telegram(
    text: str, *, token: str, chat_id: str, dry_run: bool = False
) -> bool:
    """Envía un mensaje (HTML) por Telegram. Devuelve True si se envió/imprimió.

    Si ``dry_run`` o no hay token, imprime por consola en lugar de enviar.
    """
    if dry_run or not token:
        print("─── [DRY-RUN Telegram] ───")
        print(text)
        print("──────────────────────────")
        return True

    # Import perezoso: solo se necesita aiogram en envío real.
    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode

    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    try:
        for chunk in _split_message(text):
            await bot.send_message(chat_id, chunk)
        return True
    finally:
        await bot.session.close()


def send_telegram_sync(text: str, *, token: str, chat_id: str, dry_run: bool = False) -> bool:
    """Wrapper síncrono para usar desde scripts no-async."""
    return asyncio.run(send_telegram(text, token=token, chat_id=chat_id, dry_run=dry_run))
