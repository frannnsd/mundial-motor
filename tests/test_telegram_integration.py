"""Test de integración del envío por Telegram.

Levanta un servidor HTTP que imita la Bot API de Telegram y manda un mensaje real
a través de TODO el stack de aiogram (serialización + HTTP + parseo de respuesta).
Prueba que el código ENVÍA bien de punta a punta — lo único que cambia con el token
real es el destino (api.telegram.org en vez de localhost).
"""

from __future__ import annotations

import pytest_asyncio
from aiohttp import web

from mundial_bot.notify.telegram_bot import send_telegram, verify_connection

TOKEN = "123456:TESTTOKEN"
CHAT_ID = "555"


@pytest_asyncio.fixture
async def mock_telegram():
    """Servidor que imita la Bot API. Devuelve (base_url, lista_de_mensajes_enviados)."""
    sent: list[dict] = []

    async def get_me(_request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "result": {"id": 42, "is_bot": True, "first_name": "TestBot",
                       "username": "mundial_test_bot"},
        })

    async def send_message(request: web.Request) -> web.Response:
        data = await request.post()
        text = data.get("text", "")
        sent.append({"chat_id": data.get("chat_id"), "text": text})
        return web.json_response({
            "ok": True,
            "result": {"message_id": 1, "date": 1700000000,
                       "chat": {"id": int(CHAT_ID), "type": "private"}, "text": text},
        })

    app = web.Application()
    app.router.add_route("*", "/bot{token}/getMe", get_me)
    app.router.add_route("*", "/bot{token}/sendMessage", send_message)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", sent
    finally:
        await runner.cleanup()


async def test_verify_connection_envia_mensaje_real(mock_telegram):
    base, sent = mock_telegram

    ok, message = await verify_connection(TOKEN, CHAT_ID, api_base=base)

    assert ok is True
    assert "mundial_test_bot" in message      # validó el token vía getMe
    assert len(sent) == 1                       # mandó el mensaje de prueba
    assert "Conexión OK" in sent[0]["text"]
    assert sent[0]["chat_id"] == CHAT_ID


async def test_send_telegram_viaja_por_la_red(mock_telegram):
    base, sent = mock_telegram

    ok = await send_telegram("hola mundial ⚽", token=TOKEN, chat_id=CHAT_ID, api_base=base)

    assert ok is True
    assert sent[-1]["text"] == "hola mundial ⚽"


async def test_verify_connection_sin_token_no_toca_la_red():
    ok, message = await verify_connection("", CHAT_ID)
    assert ok is False
    assert "TELEGRAM_BOT_TOKEN" in message
