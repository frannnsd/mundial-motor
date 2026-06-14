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

from mundial_bot.collectors.statsbomb_stats import EVENTS_CACHE, load_events  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402
from mundial_bot.models.cards_model import CardsModel  # noqa: E402
from mundial_bot.models.corners_model import CornersModel  # noqa: E402
from mundial_bot.notify.scheduler import start_daily_scheduler  # noqa: E402
from mundial_bot.notify.telegram_bot import send_telegram_sync  # noqa: E402
from mundial_bot.pipeline import build_models  # noqa: E402
from mundial_bot.report import build_match_report, format_match_reports  # noqa: E402
from mundial_bot.value.odds import load_sample  # noqa: E402
from mundial_bot.value.team_aliases import normalize_team  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_odds.json"


def build_report_message(date_str: str) -> str:
    """Entrena los modelos, arma los reportes de los partidos y devuelve el mensaje."""
    models = build_models()

    corners = cards = None
    if EVENTS_CACHE.exists():
        ev = load_events(build_if_missing=False)
        corners = CornersModel.from_events(ev)
        cards = CardsModel.from_events(ev)

    # TODO: con API-Football, reemplazar el sample por los fixtures reales del día
    # (y pasar el árbitro asignado a cada partido para el modelo de tarjetas).
    matches = load_sample(SAMPLE)
    reports = [
        build_match_report(
            normalize_team(m.home_team),
            normalize_team(m.away_team),
            elo=models.elo,
            goals=models.goals,
            corners=corners,
            cards=cards,
            neutral=True,
            match_name=m.match,
        )
        for m in matches
    ]
    return format_match_reports(reports, date_str=date_str)


def run_once() -> None:
    settings = get_settings()
    print("Entrenando modelos y armando reportes por partido...")
    message = build_report_message(datetime.now().strftime("%d/%m/%Y"))
    send_telegram_sync(
        message,
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        dry_run=not settings.has_telegram,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicciones multi-mercado por partido")
    parser.add_argument("--schedule", action="store_true", help="enviar a diario")
    args = parser.parse_args()

    if args.schedule:
        settings = get_settings()
        start_daily_scheduler(run_once, hour=settings.daily_picks_hour, timezone=settings.timezone)
    else:
        run_once()


if __name__ == "__main__":
    main()
