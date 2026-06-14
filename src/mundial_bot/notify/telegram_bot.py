"""Envío de mensajes por Telegram con aiogram — Agente 5.

El envío real necesita TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID (van en .env).
La función es async porque aiogram lo es; el pipeline la corre con asyncio.run.

Modo dry-run: si no hay token, imprime el mensaje por consola en vez de enviarlo,
así se puede probar el flujo completo sin credenciales.

`verify_connection` valida token + chat_id con mensajes de error claros (para el
script de setup `scripts/test_telegram.py`).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Límite de Telegram por mensaje (caracteres). Partimos si excede.
TELEGRAM_MAX_CHARS = 4096


def _split_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Parte un mensaje largo en trozos <= limit, respetando saltos de línea.

    Una línea individual más larga que el límite se parte a la fuerza (en vez de
    dejar pasar un trozo que supere el límite de Telegram).
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # Línea sola más larga que el límite → se corta a la fuerza.
        if len(line) + 1 > limit:
            if current:
                chunks.append(current.rstrip("\n"))
                current = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current.rstrip("\n"))
            current = ""
        current += line + "\n"

    if current.strip():
        chunks.append(current.rstrip("\n"))
    return chunks


def _make_bot(token: str, api_base: str | None = None):
    """Crea el Bot de aiogram. ``api_base`` permite apuntar a un servidor de prueba."""
    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode

    session = None
    if api_base:
        from aiogram.client.session.aiohttp import AiohttpSession
        from aiogram.client.telegram import TelegramAPIServer

        session = AiohttpSession(api=TelegramAPIServer.from_base(api_base))
    return Bot(token, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


async def send_telegram(
    text: str,
    *,
    token: str,
    chat_id: str,
    dry_run: bool = False,
    api_base: str | None = None,
) -> bool:
    """Envía un mensaje (HTML) por Telegram. Devuelve True si se envió/imprimió.

    Si ``dry_run`` o no hay token, imprime por consola en lugar de enviar.
    Maneja RetryAfter (flood control) reintentando una vez.
    """
    if dry_run or not token:
        print("─── [DRY-RUN Telegram] ───")
        print(text)
        print("──────────────────────────")
        return True

    from aiogram.exceptions import TelegramRetryAfter

    bot = _make_bot(token, api_base)
    try:
        for chunk in _split_message(text):
            try:
                await bot.send_message(chat_id, chunk)
            except TelegramRetryAfter as e:
                await asyncio.sleep(e.retry_after + 1)
                await bot.send_message(chat_id, chunk)
        return True
    except Exception:
        logger.exception("Error enviando mensaje por Telegram")
        return False
    finally:
        await bot.session.close()


def send_telegram_sync(text: str, *, token: str, chat_id: str, dry_run: bool = False) -> bool:
    """Wrapper síncrono para usar desde scripts no-async."""
    return asyncio.run(send_telegram(text, token=token, chat_id=chat_id, dry_run=dry_run))


async def verify_connection(
    token: str, chat_id: str, *, api_base: str | None = None
) -> tuple[bool, str]:
    """Valida el token y el chat_id enviando un mensaje de prueba.

    Devuelve (ok, mensaje_legible). Pensado para el script de setup.
    """
    if not token:
        return False, "Falta TELEGRAM_BOT_TOKEN en el .env."
    if not chat_id:
        return False, "Falta TELEGRAM_CHAT_ID en el .env."

    from aiogram.exceptions import (
        TelegramBadRequest,
        TelegramUnauthorizedError,
    )

    bot = _make_bot(token, api_base)
    try:
        me = await bot.get_me()
        await bot.send_message(
            chat_id,
            "✅ <b>Conexión OK</b> — el Mundial Value Bot ya te puede mandar las predicciones.",
        )
        return True, f"Conectado como @{me.username}. Mensaje de prueba enviado al chat {chat_id}."
    except TelegramUnauthorizedError:
        return False, "Token inválido. Revisá TELEGRAM_BOT_TOKEN (lo da @BotFather)."
    except TelegramBadRequest as e:
        return False, (
            f"chat_id inválido o el bot no puede escribirte ({e.message}). "
            "Abrí un chat con tu bot y mandale /start, y verificá TELEGRAM_CHAT_ID "
            "(lo da @userinfobot)."
        )
    except Exception as e:  # noqa: BLE001
        return False, f"Error inesperado: {e}"
    finally:
        await bot.session.close()


def verify_connection_sync(token: str, chat_id: str) -> tuple[bool, str]:
    """Wrapper síncrono de verify_connection."""
    return asyncio.run(verify_connection(token, chat_id))
