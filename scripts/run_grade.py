"""Califica las predicciones de partidos ya terminados y manda el balance por Telegram.

Uso:  python scripts/run_grade.py   (correr después de que jueguen los partidos)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.tracking import PredictionStore, format_balance, grade_pending  # noqa: E402


def main() -> None:
    settings = get_settings()
    if not settings.has_api_football:
        print("Falta API_FOOTBALL_KEY en .env.")
        sys.exit(1)

    print("Calificando predicciones de partidos terminados...")
    graded = grade_pending(settings.api_football_key)
    print(f"Predicciones calificadas en esta corrida: {graded}")

    with PredictionStore() as store:
        message = format_balance(store.balance())

    send_telegram_sync(
        message,
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        dry_run=not settings.has_telegram,
    )


if __name__ == "__main__":
    main()
