"""Modelo de córners totales por partido.

Modelo multiplicativo de ataque/defensa (como Dixon-Coles pero para córners):
  córners_esperados_local = córners_a_favor(local) × córners_en_contra(visita) / promedio_liga

Suma local + visita → córners totales esperados → over/under por línea (Poisson).
Equipos sin datos caen al promedio de la liga (predicción robusta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mundial_bot.models.count_market import (
    CORNER_LINES,
    best_line,
    over_under,
    shrink,
    weighted_means,
)

# Búsqueda de la calibración (factor sobre el total esperado) y mínimo de muestra.
_CALIB_GRID = np.arange(0.85, 1.251, 0.05)
_CALIB_MIN_MATCHES = 60
_CALIB_BOUNDS = (0.85, 1.25)


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
    calibration: float = 1.0        # corrige el sesgo del total (auto-calibrado con datos)

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> CornersModel:
        """Construye el modelo desde el DataFrame de estadísticas (StatsBomb o API-Football)."""
        # Media ponderada por recencia (partidos recientes pesan más) + shrinkage.
        means, eff = weighted_means(events, ["corners_for", "corners_against"])
        for_w, against_w = means["corners_for"], means["corners_against"]
        league_avg = float(events["corners_for"].mean())
        team_for = {t: shrink(for_w[t], eff[t], league_avg) for t in for_w}
        team_against = {t: shrink(against_w[t], eff[t], league_avg) for t in against_w}

        # Sobre-dispersión de los córners totales por partido (para Negative Binomial).
        per_match = events.drop_duplicates("match_id") if "match_id" in events else events
        totals = per_match["corners_for"] + per_match["corners_against"]
        mean_t = float(totals.mean())
        dispersion = max(1.0, float(totals.var()) / mean_t) if mean_t > 0 else 1.0

        model = cls(
            team_for=team_for, team_against=team_against,
            league_avg=league_avg, dispersion=dispersion,
        )
        model.calibration = model._fit_calibration(events)
        return model

    def _expected_side(self, attacker: str, defender: str) -> float:
        att = self.team_for.get(attacker, self.league_avg)
        deff = self.team_against.get(defender, self.league_avg)
        if self.league_avg <= 0:
            return att
        return att * deff / self.league_avg

    def _base_total(self, home: str, away: str) -> float:
        return self._expected_side(home, away) + self._expected_side(away, home)

    def _fit_calibration(self, events: pd.DataFrame) -> float:
        """Calibra el total contra los over reales: elige el factor que mejor matchea
        la frecuencia real de over en la línea central. Auto-calibración con datos."""
        needed = {"match_id", "team", "opponent", "corners_for", "corners_against"}
        if not needed.issubset(events.columns):
            return 1.0
        rows = []
        for _, g in events.groupby("match_id"):
            if "is_home" in g.columns:
                home_rows = g[g["is_home"] == 1]
                r = home_rows.iloc[0] if len(home_rows) else g.iloc[0]
            else:
                r = g.iloc[0]
            opp = r.get("opponent")
            if not isinstance(opp, str):
                continue
            rows.append((r["team"], opp, float(r["corners_for"] + r["corners_against"])))
        if len(rows) < _CALIB_MIN_MATCHES:
            return 1.0
        base = np.array([self._base_total(h, a) for h, a, _ in rows])
        actual = np.array([t for _, _, t in rows])
        line = round(float(np.median(actual))) + 0.5
        real_over = float((actual > line).mean())
        best_f, best_err = 1.0, 1e9
        for f in _CALIB_GRID:
            tot = base * f
            p_over = np.array([over_under(t, line, variance=t * self.dispersion)[0] for t in tot])
            err = abs(float(p_over.mean()) - real_over)
            if err < best_err:
                best_f, best_err = float(f), err
        return min(max(best_f, _CALIB_BOUNDS[0]), _CALIB_BOUNDS[1])

    def predict(self, home: str, away: str) -> CornersPrediction:
        home_c = self._expected_side(home, away)
        away_c = self._expected_side(away, home)
        total = (home_c + away_c) * self.calibration
        variance = total * self.dispersion
        line = best_line(total, CORNER_LINES, variance=variance)
        p_over, p_under = over_under(total, line, variance=variance)
        return CornersPrediction(
            home_corners=home_c, away_corners=away_c, total=total,
            line=line, p_over=p_over, p_under=p_under,
        )
