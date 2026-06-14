"""Entrypoint de producción: genera la cartilla del día y la manda por Telegram.

Uso:
  python scripts/run_daily.py            # corre una vez (usa cuotas reales si hay key)
  python scripts/run_daily.py --sample   # demo offline con cuotas de ejemplo
  python scripts/run_daily.py --schedule # queda corriendo y envía a diario

Sin claves cargadas en .env, cae automáticamente a modo demo (dry-run): imprime la
cartilla por consola en vez de enviarla.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.scheduler import start_daily_scheduler  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.pipeline import build_models, run_pipeline  # noqa: E402
from mundial_bot.value.odds import load_sample  # noqa: E402

SAMPLE_ODDS = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_odds.json"


def run_once(*, use_sample: bool) -> None:
    settings = get_settings()
    print("Entrenando modelos (Elo + Dixon-Coles) sobre el histórico...")
    models = build_models()

    if use_sample or not settings.has_odds_api:
        if not settings.has_odds_api and not use_sample:
            print("⚠️  Sin ODDS_API_KEY → modo demo con cuotas de ejemplo.")
        matches = load_sample(SAMPLE_ODDS)
    else:
        matches = None  # el pipeline trae cuotas reales de The Odds API

    result = run_pipeline(settings=settings, models=models, matches=matches,
                          record_ledger=True)

    print(f"\nPartidos analizados: {result.n_matches} · "
          f"Value picks: {len(result.value_picks)} · "
          f"Combinadas: {len(result.parlays)}\n")

    send_telegram_sync(
        result.message,
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        dry_run=not settings.has_telegram,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartilla diaria del Mundial Value Bot")
    parser.add_argument("--sample", action="store_true", help="usar cuotas de ejemplo (offline)")
    parser.add_argument("--schedule", action="store_true", help="correr en modo programado diario")
    args = parser.parse_args()

    if args.schedule:
        settings = get_settings()
        start_daily_scheduler(
            lambda: run_once(use_sample=args.sample),
            hour=settings.daily_picks_hour,
            timezone=settings.timezone,
        )
    else:
        run_once(use_sample=args.sample)


if __name__ == "__main__":
    main()
