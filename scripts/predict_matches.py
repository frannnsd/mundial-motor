"""Reporte de predicciones multi-mercado por partido.

Muestra, por partido: ganador, goles, córners, tarjetas y ambos marcan, cada uno
con su cuota justa para comparar contra tu casa.

Uso:  python scripts/predict_matches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial_bot.collectors.statsbomb_stats import EVENTS_CACHE, load_events  # noqa: E402
from mundial_bot.models.cards_model import CardsModel  # noqa: E402
from mundial_bot.models.corners_model import CornersModel  # noqa: E402
from mundial_bot.pipeline import build_models  # noqa: E402
from mundial_bot.report import build_match_report, format_match_reports  # noqa: E402
from mundial_bot.value.odds import load_sample  # noqa: E402
from mundial_bot.value.team_aliases import normalize_team  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_odds.json"


def main() -> None:
    print("Entrenando modelos de goles/ganador (Elo + Dixon-Coles)...")
    models = build_models()

    corners = cards = None
    if EVENTS_CACHE.exists():
        print("Cargando modelos de córners/tarjetas (StatsBomb)...")
        ev = load_events(build_if_missing=False)
        corners = CornersModel.from_events(ev)
        cards = CardsModel.from_events(ev)
    else:
        print("(Cache de StatsBomb aún no lista → muestro goles+ganador. "
              "Corré scripts/fetch_statsbomb.py para sumar córners/tarjetas.)")

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
    print()
    print(format_match_reports(reports, date_str="14/06/2026"))


if __name__ == "__main__":
    main()
