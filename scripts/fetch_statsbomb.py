"""Baja y cachea los eventos de córners/tarjetas/faltas de StatsBomb.

Uso:  python scripts/fetch_statsbomb.py   (tarda varios minutos, una sola vez)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mundial_bot.collectors.statsbomb_stats import build_cache  # noqa: E402


def main() -> None:
    print("Bajando eventos de StatsBomb (WC, Euro, Copa América, AFCON)... puede tardar.")
    path = build_cache()
    df = pd.read_csv(path)
    print(f"\nListo: {len(df)} filas (equipo-partido) cacheadas en {path}")
    print(f"Equipos: {df['team'].nunique()} · Partidos: {df['match_id'].nunique()} · "
          f"Árbitros: {df['referee'].nunique()}")
    print(f"Córners promedio/equipo: {df['corners_for'].mean():.1f} · "
          f"Tarjetas: {df['cards'].mean():.1f} · Faltas: {df['fouls'].mean():.1f}")


if __name__ == "__main__":
    main()
