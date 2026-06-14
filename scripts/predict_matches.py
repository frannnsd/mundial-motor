"""Reporte de predicciones multi-mercado por partido (el predictor principal).

Por cada partido muestra lo más probable en: ganador, goles, córners, tarjetas y
ambos marcan — cada uno con su cuota justa para comparar contra tu casa
(Bet365/bplay/Stake). Lo manda por Telegram (dry-run a consola si no hay token).

Uso:
  python scripts/predict_matches.py            # corre una vez
  python scripts/predict_matches.py --schedule # envía a diario a la hora configurada
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.brain import build_today_message, load_brain  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.notify.scheduler import start_daily_scheduler  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.report import build_match_report, format_match_reports  # noqa: E402
from mundial_bot.value.odds import load_sample  # noqa: E402
from mundial_bot.value.team_aliases import normalize_team  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_odds.json"


def build_report_message(date_str: str) -> str:
    """Arma la cartilla del día. Con key real usa los fixtures de hoy (y loguea); sino, ejemplo."""
    settings = get_settings()
    brain = load_brain()

    if settings.has_api_football or settings.has_football_data:
        return build_today_message(brain, settings, date_str=date_str)

    # Sin keys de fixtures → lista de ejemplo (modo dev).
    print("⚠️  Sin API_FOOTBALL_KEY → uso la lista de ejemplo.")
    matches = load_sample(SAMPLE)
    reports = [
        build_match_report(
            normalize_team(m.home_team), normalize_team(m.away_team),
            elo=brain.models.elo, goals=brain.models.goals,
            corners=brain.corners, cards=brain.cards,
            neutral=True, match_name=m.match,
        )
        for m in matches
    ]
    return format_match_reports(reports, date_str=date_str)


def run_once(*, dry: bool = False) -> None:
    settings = get_settings()
    print("Entrenando modelos y armando reportes por partido...")
    message = build_report_message(datetime.now().strftime("%d/%m/%Y"))
    ok = send_telegram_sync(
        message,
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        dry_run=dry or not settings.has_telegram,
    )
    if settings.has_telegram and not dry:
        print("✅ Cartilla enviada a tu Telegram." if ok
              else "❌ No se pudo enviar (revisá token/chat_id).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicciones multi-mercado por partido")
    parser.add_argument("--schedule", action="store_true", help="enviar a diario")
    parser.add_argument("--dry", action="store_true", help="imprimir en consola, no enviar")
    args = parser.parse_args()

    if args.schedule:
        settings = get_settings()
        start_daily_scheduler(
            lambda: run_once(dry=args.dry),
            hour=settings.daily_picks_hour,
            timezone=settings.timezone,
        )
    else:
        run_once(dry=args.dry)


if __name__ == "__main__":
    main()
