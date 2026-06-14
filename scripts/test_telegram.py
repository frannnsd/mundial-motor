"""Prueba la conexión con Telegram usando el token y chat_id del .env.

Uso:  python scripts/test_telegram.py

Si todo está bien, te llega un mensaje de prueba a tu Telegram.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.telegram_bot import verify_connection_sync  # noqa: E402


def main() -> None:
    settings = get_settings()
    print("Verificando conexión con Telegram...\n")
    ok, message = verify_connection_sync(settings.telegram_bot_token, settings.telegram_chat_id)

    if ok:
        print(f"✅ {message}")
        print("\n¡Listo! Revisá tu Telegram: te tiene que haber llegado el mensaje de prueba.")
        print("Ahora podés correr:  python scripts/predict_matches.py --schedule")
        return

    print(f"❌ {message}\n")
    print("Cómo conseguir las credenciales:")
    print("  1. Hablá con @BotFather  →  /newbot  →  copiá el TOKEN.")
    print("  2. Hablá con @userinfobot →  copiá tu ID numérico (TELEGRAM_CHAT_ID).")
    print("  3. Abrí un chat con tu bot nuevo y mandale /start (si no, no te puede escribir).")
    print("  4. Pegá ambos en el archivo .env y volvé a correr este script.")
    sys.exit(1)


if __name__ == "__main__":
    main()
