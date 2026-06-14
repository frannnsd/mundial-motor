"""Conecta y prueba Telegram en un solo comando.

Uso:
  # Opción 1 — pasás las credenciales y se guardan en .env + se prueban:
  python scripts/test_telegram.py --token 8123:AAH... --chat-id 5551234

  # Opción 2 — ya cargaste el .env a mano y solo querés probar:
  python scripts/test_telegram.py

Si todo está bien, te llega un mensaje de prueba a tu Telegram.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.config import PROJECT_ROOT, get_settings  # noqa: E402
from mundial_bot.notify.telegram_bot import verify_connection_sync  # noqa: E402

ENV_PATH = PROJECT_ROOT / ".env"


def _persist_credentials(token: str, chat_id: str) -> None:
    """Escribe el token y el chat_id en el .env, preservando el resto."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    updated = {"TELEGRAM_BOT_TOKEN": False, "TELEGRAM_CHAT_ID": False}
    new_values = {"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": chat_id}

    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in new_values:
            out.append(f"{key}={new_values[key]}")
            updated[key] = True
        else:
            out.append(line)
    for key, done in updated.items():
        if not done:
            out.append(f"{key}={new_values[key]}")

    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Credenciales guardadas en {ENV_PATH}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Conecta y prueba Telegram")
    parser.add_argument("--token", help="token de @BotFather")
    parser.add_argument("--chat-id", dest="chat_id", help="tu chat id de @userinfobot")
    args = parser.parse_args()

    if args.token and args.chat_id:
        _persist_credentials(args.token.strip(), args.chat_id.strip())

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
    print("  4. Volvé a correr:  python scripts/test_telegram.py --token <TOKEN> --chat-id <ID>")
    sys.exit(1)


if __name__ == "__main__":
    main()
