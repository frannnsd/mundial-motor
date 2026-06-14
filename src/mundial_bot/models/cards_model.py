"""Modelo de tarjetas totales por partido.

El driver más fuerte de las tarjetas NO son los equipos sino el **árbitro**: su
promedio de tarjetas por partido es el mejor predictor. Combinamos:
  - severidad del árbitro (tarjetas totales promedio en sus partidos)
  - disciplina de los equipos (tarjetas promedio que reciben)
  - importancia del partido (knockout > fase de grupos → más tarjetas)

esperadas = importancia × (½ · base_equipos + ½ · base_árbitro)

Datos faltantes (equipo o árbitro desconocido) caen al promedio de la liga.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mundial_bot.models.count_market import CARD_LINES, closest_line, over_under

# Multiplicador de importancia (calibrable).
IMPORTANCE_GROUP = 1.0
IMPORTANCE_KNOCKOUT = 1.15


@dataclass
class CardsPrediction:
    total: float
    line: float
    p_over: float
    p_under: float
    referee: str | None


@dataclass
class CardsModel:
    team_cards: dict[str, float]     # tarjetas promedio que recibe cada equipo
    referee_cards: dict[str, float]  # tarjetas totales promedio del árbitro
    league_team_avg: float           # tarjetas por equipo promedio (liga)
    league_ref_avg: float            # tarjetas totales por partido promedio (liga)
    dispersion: float = 1.0          # factor de Fano (var/media) de las tarjetas totales

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> CardsModel:
        """Construye el modelo desde el DataFrame de estadísticas (StatsBomb o API-Football)."""
        team_cards = events.groupby("team")["cards"].mean().to_dict()
        league_team_avg = float(events["cards"].mean())

        per_match = events.groupby("match_id").agg(
            total_cards=("cards", "sum"), referee=("referee", "first")
        )
        referee_cards = per_match.groupby("referee")["total_cards"].mean().to_dict()
        league_ref_avg = float(per_match["total_cards"].mean())

        mean_c = float(per_match["total_cards"].mean())
        dispersion = max(1.0, float(per_match["total_cards"].var()) / mean_c) if mean_c > 0 else 1.0

        return cls(
            team_cards=team_cards,
            referee_cards=referee_cards,
            league_team_avg=league_team_avg,
            league_ref_avg=league_ref_avg,
            dispersion=dispersion,
        )

    def predict(
        self,
        home: str,
        away: str,
        *,
        referee: str | None = None,
        knockout: bool = False,
    ) -> CardsPrediction:
        team_base = (
            self.team_cards.get(home, self.league_team_avg)
            + self.team_cards.get(away, self.league_team_avg)
        )
        if referee and referee in self.referee_cards:
            ref_base = self.referee_cards[referee]
        else:
            ref_base = self.league_ref_avg
        importance = IMPORTANCE_KNOCKOUT if knockout else IMPORTANCE_GROUP

        total = importance * (0.5 * team_base + 0.5 * ref_base)
        line = closest_line(total, CARD_LINES)
        p_over, p_under = over_under(total, line, variance=total * self.dispersion)
        return CardsPrediction(
            total=total, line=line, p_over=p_over, p_under=p_under, referee=referee
        )
