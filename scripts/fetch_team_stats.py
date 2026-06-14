"""Baja la forma reciente (córners/tarjetas/tiros/faltas) de los equipos del Mundial.

Uso:  python scripts/fetch_team_stats.py   (cientos de requests, ~min, una vez por día)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mundial_bot.collectors.team_stats import build_cache  # noqa: E402
from mundial_bot.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    settings = get_settings()
    if not settings.has_api_football:
        print("Falta API_FOOTBALL_KEY en .env.")
        sys.exit(1)

    print("Bajando forma reciente de los equipos del Mundial (API-Football Pro)...")
    path = build_cache(settings.api_football_key, last=12)
    df = pd.read_csv(path)
    print(f"\nListo: {len(df)} filas (equipo-partido) en {path}")
    print(f"Equipos: {df['team'].nunique()} · Partidos: {df['match_id'].nunique()} · "
          f"Árbitros: {df['referee'].nunique()}")
    print(f"Córners prom/equipo: {df['corners_for'].mean():.1f} · "
          f"Tarjetas: {df['cards'].mean():.1f} · Tiros: {df['shots'].mean():.1f} · "
          f"Faltas: {df['fouls'].mean():.1f}")


if __name__ == "__main__":
    main()
