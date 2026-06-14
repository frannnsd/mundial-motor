"""Backtest de córners y tarjetas sobre la forma reciente (validar edge).

Uso:  python scripts/backtest_counts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mundial_bot.backtest.count_backtest import backtest_cards, backtest_corners  # noqa: E402
from mundial_bot.collectors.team_stats import load_team_stats  # noqa: E402


def main() -> None:
    df = load_team_stats()
    if "date" not in df.columns:
        print("La cache de team_stats no tiene fecha. Corré fetch_team_stats.py de nuevo.")
        sys.exit(1)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Backtest sobre {df['match_id'].nunique()} partidos.\n")
    print(backtest_corners(df).summary())
    print(backtest_cards(df).summary())


if __name__ == "__main__":
    main()
