"""Actualización diaria completa (el loop que se autoalimenta).

Pensado para correr una vez por día (cron / Railway):
  1. Refresca la forma reciente de los equipos (API-Football).
  2. Actualiza los resultados del Mundial → el Elo se autoalimenta.
  3. Califica las predicciones de los partidos que ya terminaron.
  4. Manda el balance + las predicciones de hoy por Telegram.

Uso:  python scripts/run_daily_update.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.brain import build_today_message, load_brain  # noqa: E402
from mundial_bot.collectors.team_stats import build_cache as refresh_team_stats  # noqa: E402
from mundial_bot.collectors.wc_results import build_cache as refresh_wc_results  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.tracking import PredictionStore, format_balance, grade_pending  # noqa: E402


def main() -> None:
    settings = get_settings()
    if not settings.has_api_football:
        print("Falta API_FOOTBALL_KEY en .env.")
        sys.exit(1)
    key = settings.api_football_key

    print("1/4 Refrescando forma reciente de los equipos...")
    refresh_team_stats(key, last=12)

    print("2/4 Actualizando resultados del Mundial (el Elo se autoalimenta)...")
    wc = refresh_wc_results(key)
    print(f"     {len(wc)} partidos del Mundial finalizados en el histórico del Elo.")

    print("3/4 Calificando predicciones de partidos terminados...")
    graded = grade_pending(key)
    print(f"     {graded} predicciones calificadas.")

    print("4/4 Armando y enviando balance + predicciones de hoy...")
    with PredictionStore() as store:
        balance = format_balance(store.balance())
    brain = load_brain()
    today = build_today_message(brain, settings, date_str=datetime.now().strftime("%d/%m/%Y"))

    send_telegram_sync(
        balance + "\n\n" + today,
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        dry_run=not settings.has_telegram,
    )
    print("✅ Listo.")


if __name__ == "__main__":
    main()
