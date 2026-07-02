"""Modelo de tiros al arco (shots on goal) totales por partido.

Mismo enfoque que córners: modelo multiplicativo de ataque/defensa
  tiros_al_arco_esperados(local) = a_favor(local) × en_contra(visita) / promedio_liga
Suma local + visita → total esperado → over/under por línea (Negative Binomial).
Se auto-calibra contra la frecuencia real de over (como córners).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mundial_bot.models.count_market import best_line, over_under, shrink, weighted_means

SHOT_LINES = (5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
_CALIB_GRID = np.arange(0.85, 1.251, 0.05)
_CALIB_MIN_MATCHES = 60
_CALIB_BOUNDS = (0.85, 1.25)


@dataclass
class ShotsPrediction:
    home_shots: float
    away_shots: float
    total: float
    line: float
    p_over: float
    p_under: float


@dataclass
class ShotsModel:
    team_for: dict[str, float]      # tiros al arco a favor promedio por equipo
    team_against: dict[str, float]  # tiros al arco en contra promedio por equipo
    league_avg: float
    dispersion: float = 1.0
    calibration: float = 1.0

    @classmethod
    def from_events(
        cls, events: pd.DataFrame, *, as_of: pd.Timestamp | str | None = None
    ) -> ShotsModel | None:
        """Construye el modelo si están las columnas sot_for/sot_against; si no, None.

        ``as_of`` (POINT-IN-TIME): descarta partidos con fecha >= kickoff. ``None`` =
        comportamiento histórico (path live intacto).
        """
        if not {"sot_for", "sot_against"}.issubset(events.columns):
            return None
        if as_of is not None and "date" in events.columns:
            events = events.loc[
                pd.to_datetime(events["date"], errors="coerce") < pd.Timestamp(as_of)
            ].copy()
        ev = events.dropna(subset=["sot_for", "sot_against"])
        if ev.empty or float(ev["sot_for"].sum()) <= 0:
            return None
        means, eff = weighted_means(ev, ["sot_for", "sot_against"], as_of=as_of)
        for_w, against_w = means["sot_for"], means["sot_against"]
        league_avg = float(ev["sot_for"].mean())
        team_for = {t: shrink(for_w[t], eff[t], league_avg) for t in for_w}
        team_against = {t: shrink(against_w[t], eff[t], league_avg) for t in against_w}

        per_match = ev.drop_duplicates("match_id") if "match_id" in ev else ev
        totals = per_match["sot_for"] + per_match["sot_against"]
        mean_t = float(totals.mean())
        dispersion = max(1.0, float(totals.var()) / mean_t) if mean_t > 0 else 1.0

        model = cls(team_for=team_for, team_against=team_against,
                    league_avg=league_avg, dispersion=dispersion)
        model.calibration = model._fit_calibration(ev)
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
        """Igual que córners: factor que mejor matchea la frecuencia real de over."""
        if "match_id" not in events.columns:
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
            rows.append((r["team"], opp, float(r["sot_for"] + r["sot_against"])))
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

    def predict(self, home: str, away: str) -> ShotsPrediction:
        home_s = self._expected_side(home, away)
        away_s = self._expected_side(away, home)
        total = (home_s + away_s) * self.calibration
        variance = total * self.dispersion
        line = best_line(total, SHOT_LINES, variance=variance)
        p_over, p_under = over_under(total, line, variance=variance)
        return ShotsPrediction(
            home_shots=home_s, away_shots=away_s, total=total,
            line=line, p_over=p_over, p_under=p_under,
        )
