"""Demo: entrena el Elo con el histórico real y muestra rankings + predicciones.

Uso:  python scripts/demo_elo.py
"""

from __future__ import annotations

from mundial_bot.collectors.results import load_results
from mundial_bot.models.elo_model import EloModel


def main() -> None:
    print("Cargando histórico internacional (martj42)...")
    df = load_results(since="1990-01-01")
    print(f"  {len(df):,} partidos desde 1990.\n")

    print("Entrenando Elo internacional...")
    model = EloModel().fit(df)
    print(f"  {len(model.ratings):,} selecciones rankeadas.\n")

    print("=== TOP 20 RANKING ELO (a hoy) ===")
    for i, (team, rating) in enumerate(model.rankings(top=20), 1):
        print(f"{i:>2}. {team:<22} {rating:7.1f}")

    print("\n=== PREDICCIONES (cancha neutral, estilo Mundial) ===")
    duelos = [
        ("Argentina", "Brazil"),
        ("France", "Spain"),
        ("Argentina", "Bolivia"),
        ("England", "Germany"),
    ]
    for home, away in duelos:
        p = model.predict(home, away, neutral=True)
        print(
            f"{home} vs {away:<10} | "
            f"{home[:3]} {p.home:5.1%}  X {p.draw:5.1%}  {away[:3]} {p.away:5.1%}"
        )


if __name__ == "__main__":
    main()
