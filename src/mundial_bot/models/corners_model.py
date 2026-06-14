"""Modelo de córners totales por partido.

Modelo multiplicativo de ataque/defensa (como Dixon-Coles pero para córners):
  córners_esperados_local = córners_a_favor(local) × córners_en_contra(visita) / promedio_liga

Suma local + visita → córners totales esperados → over/under por línea (Poisson).
Equipos sin datos caen al promedio de la liga (predicción robusta).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mundial_bot.models.count_market import CORNER_LINES, best_line, over_under, shrink


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
    dispersion: float = 1.0         # factor de Fano (var/media) de los córners totales

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> CornersModel:
        """Construye el modelo desde el DataFrame de estadísticas (StatsBomb o API-Football)."""
        grp = events.groupby("team")
        counts = grp.size()
        for_mean = grp["corners_for"].mean()
        against_mean = grp["corners_against"].mean()
        league_avg = float(events["corners_for"].mean())

        # Shrinkage: equipos con pocos partidos tiran hacia la media de la liga.
        team_for = {t: shrink(for_mean[t], counts[t], league_avg) for t in for_mean.index}
        team_against = {
            t: shrink(against_mean[t], counts[t], league_avg) for t in against_mean.index
        }

        # Sobre-dispersión de los córners totales por partido (para Negative Binomial).
        per_match = events.drop_duplicates("match_id") if "match_id" in events else events
        totals = per_match["corners_for"] + per_match["corners_against"]
        mean_t = float(totals.mean())
        dispersion = max(1.0, float(totals.var()) / mean_t) if mean_t > 0 else 1.0

        return cls(
            team_for=team_for, team_against=team_against,
            league_avg=league_avg, dispersion=dispersion,
        )

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
        variance = total * self.dispersion
        line = best_line(total, CORNER_LINES, variance=variance)
        p_over, p_under = over_under(total, line, variance=variance)
        return CornersPrediction(
            home_corners=home_c, away_corners=away_c, total=total,
            line=line, p_over=p_over, p_under=p_under,
        )
