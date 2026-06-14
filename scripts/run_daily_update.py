"""Actualización diaria completa (refresca datos + califica + autoalimenta + envía).

Pensado para un cron diario (o ya lo corre run_bot.py solo).

Uso:  python scripts/run_daily_update.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.service import daily_cycle  # noqa: E402


def main() -> None:
    settings = get_settings()
    if not settings.has_api_football:
        print("Falta API_FOOTBALL_KEY en .env.")
        sys.exit(1)

    print("Refrescando datos + calificando + autoalimentando el cerebro...")
    message, _brain = daily_cycle(settings)
    send_telegram_sync(
        message,
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        dry_run=not settings.has_telegram,
    )
    print("✅ Listo.")


if __name__ == "__main__":
    main()
