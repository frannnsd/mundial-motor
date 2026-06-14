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
from mundial_bot.collectors.team_stats import TEAM_STATS_CACHE, load_team_stats  # noqa: E402
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


def _fetch_real_fixtures(settings):
    """Intenta traer los fixtures reales de hoy. Devuelve (lista, fuente)."""
    from datetime import date as _date

    today = _date.today().isoformat()

    # football-data.org (gratis, soporta el Mundial 2026 + árbitro).
    if settings.has_football_data:
        try:
            from mundial_bot.collectors.fixtures_fdorg import FootballDataClient

            fixtures = FootballDataClient(settings.football_data_key).get_fixtures(date=today)
            if fixtures:
                return fixtures, "football-data.org"
        except Exception as exc:  # noqa: BLE001
            print(f"   football-data.org no respondió: {exc}")

    # API-Football (OJO: el plan gratis NO da acceso a 2026).
    if settings.has_api_football:
        try:
            from mundial_bot.collectors.fixtures import FixturesClient

            fixtures = FixturesClient(settings.api_football_key).get_fixtures(date=today)
            if fixtures:
                return fixtures, "API-Football"
        except Exception as exc:  # noqa: BLE001
            print(f"   API-Football no respondió: {exc}")

    return [], None


def _match_specs(settings) -> list[tuple[str, str, str, str | None, bool]]:
    """[(home, away, nombre, árbitro, knockout)] desde una fuente real o el sample."""
    fixtures, source = _fetch_real_fixtures(settings)
    if fixtures:
        print(f"Partidos reales de hoy ({source}): {len(fixtures)}")
        return [
            (
                normalize_team(f.home_team), normalize_team(f.away_team),
                f.match, f.referee, f.knockout,
            )
            for f in fixtures
        ]

    print("⚠️  Sin fixtures reales disponibles → uso la lista de ejemplo.")
    matches = load_sample(SAMPLE)
    return [
        (normalize_team(m.home_team), normalize_team(m.away_team), m.match, None, False)
        for m in matches
    ]


def build_report_message(date_str: str) -> str:
    """Entrena los modelos, arma los reportes de los partidos y devuelve el mensaje."""
    settings = get_settings()
    models = build_models()

    # Preferir la forma reciente de API-Football (más rica) sobre StatsBomb.
    corners = cards = None
    stats_df = None
    if TEAM_STATS_CACHE.exists():
        stats_df = load_team_stats()
        print(f"Cerebro: forma reciente de API-Football ({len(stats_df)} filas)")
    elif EVENTS_CACHE.exists():
        stats_df = load_events(build_if_missing=False)
        print(f"Cerebro: StatsBomb histórico ({len(stats_df)} filas)")
    if stats_df is not None and not stats_df.empty:
        corners = CornersModel.from_events(stats_df)
        cards = CardsModel.from_events(stats_df)

    specs = _match_specs(settings)
    if not specs:
        return f"🔮 <b>PREDICCIONES — {date_str}</b>\n\nHoy no hay partidos del Mundial. 🟢"

    reports = [
        build_match_report(
            home, away,
            elo=models.elo, goals=models.goals, corners=corners, cards=cards,
            referee=referee, knockout=knockout, neutral=True, match_name=name,
        )
        for (home, away, name, referee, knockout) in specs
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
