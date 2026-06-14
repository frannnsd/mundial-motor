"""Alertas pre-partido: ~1-2h antes de cada partido manda la predicción + bajas.

Pensado para un cron cada ~30 min (o ya lo corre run_bot.py solo).

Uso:  python scripts/run_prematch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.brain import load_brain  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.service import prematch_alerts  # noqa: E402


def main() -> None:
    settings = get_settings()
    if not settings.has_telegram or not settings.has_api_football:
        print("Faltan TELEGRAM_* o API_FOOTBALL_KEY en .env.")
        sys.exit(1)

    brain = load_brain()
    messages = prematch_alerts(settings, brain)
    for msg in messages:
        send_telegram_sync(
            msg, token=settings.telegram_bot_token, chat_id=settings.telegram_chat_id
        )
    print(f"Alertas pre-partido enviadas: {len(messages)}")


if __name__ == "__main__":
    main()
