"""Modelo de córners totales por partido.

Modelo multiplicativo de ataque/defensa (como Dixon-Coles pero para córners):
  córners_esperados_local = córners_a_favor(local) × córners_en_contra(visita) / promedio_liga

Suma local + visita → córners totales esperados → over/under por línea (Poisson).
Equipos sin datos caen al promedio de la liga (predicción robusta).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mundial_bot.models.count_market import CORNER_LINES, closest_line, over_under


@dataclass
class CornersPrediction:
    home_corners: float
    away_corners: float
    total: float
    line: float          # línea más pareja
    p_over: float
    p_under: float


@dataclass
class CornersModel:
    team_for: dict[str, float]      # córners a favor promedio por equipo
    team_against: dict[str, float]  # córners en contra promedio por equipo
    league_avg: float               # córners por equipo promedio (liga)

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> CornersModel:
        """Construye el modelo desde el DataFrame de eventos de StatsBomb."""
        team_for = events.groupby("team")["corners_for"].mean().to_dict()
        team_against = events.groupby("team")["corners_against"].mean().to_dict()
        league_avg = float(events["corners_for"].mean())
        return cls(team_for=team_for, team_against=team_against, league_avg=league_avg)

    def _expected_side(self, attacker: str, defender: str) -> float:
        att = self.team_for.get(attacker, self.league_avg)
        deff = self.team_against.get(defender, self.league_avg)
        if self.league_avg <= 0:
            return att
        return att * deff / self.league_avg

    def predict(self, home: str, away: str) -> CornersPrediction:
        home_c = self._expected_side(home, away)
        away_c = self._expected_side(away, home)
        total = home_c + away_c
        line = closest_line(total, CORNER_LINES)
        p_over, p_under = over_under(total, line)
        return CornersPrediction(
            home_corners=home_c, away_corners=away_c, total=total,
            line=line, p_over=p_over, p_under=p_under,
        )
