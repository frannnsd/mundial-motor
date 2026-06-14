"""Backtest walk-forward del Elo sobre el histórico real.

Uso:  python scripts/backtest_elo.py
"""

from __future__ import annotations

from mundial_bot.backtest.walk_forward import walk_forward_elo
from mundial_bot.collectors.results import load_results


def main() -> None:
    print("Cargando histórico (martj42) desde 1990...")
    df = load_results(since="1990-01-01")
    print(f"  {len(df):,} partidos.\n")

    print("Corriendo walk-forward (calienta 1990-2014, evalua 2015 -> hoy)...")
    result = walk_forward_elo(df, start="2015-01-01")
    print("\n=== RESULTADO DEL BACKTEST (out-of-sample) ===")
    print(result.summary())
    print(
        "\nNota: una predicción 'al azar' da RPS ~0.22 y accuracy ~33%. "
        "El piso de mercado profesional ronda RPS 0.18."
    )


if __name__ == "__main__":
    main()
