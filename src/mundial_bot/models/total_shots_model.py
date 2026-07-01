"""Modelo de remates TOTALES por partido (tiros al arco + afuera).

Igual enfoque que córners/tiros al arco: modelo multiplicativo de ataque/defensa
  remates_esperados(local) = a_favor(local) × en_contra(visita) / promedio_liga

La columna `shots` del cache es el total de remates a favor de cada equipo; los
remates EN CONTRA se derivan del rival en el mismo partido (la otra fila del match).
Suma local + visita → total esperado → over/under por línea (Negative Binomial).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mundial_bot.models.count_market import shrink, weighted_means


@dataclass
class TotalShotsPrediction:
    home_shots: float
    away_shots: float
    total: float


@dataclass
class TotalShotsModel:
    team_for: dict[str, float]
    team_against: dict[str, float]
    league_avg: float
    dispersion: float = 1.0

    @classmethod
    def from_events(
        cls, events: pd.DataFrame, *, as_of: pd.Timestamp | str | None = None
    ) -> TotalShotsModel | None:
        """Construye el modelo si están las columnas necesarias; si no, None.

        ``as_of`` (POINT-IN-TIME): descarta partidos con fecha >= kickoff. ``None`` =
        comportamiento histórico (path live intacto).
        """
        need = {"shots", "match_id", "team"}
        if not need.issubset(events.columns):
            return None
        if as_of is not None and "date" in events.columns:
            events = events.loc[
                pd.to_datetime(events["date"], errors="coerce") < pd.Timestamp(as_of)
            ].copy()
        ev = events.dropna(subset=["shots"]).copy()
        if ev.empty or float(ev["shots"].sum()) <= 0:
            return None

        # Remates en contra = remates del rival en el mismo partido.
        against: dict[tuple, float] = {}
        for mid, g in ev.groupby("match_id"):
            recs = g[["team", "shots"]].to_dict("records")
            if len(recs) == 2:
                a, b = recs
                against[(mid, a["team"])] = b["shots"]
                against[(mid, b["team"])] = a["shots"]
        ev["shots_against"] = [
            against.get((m, t), np.nan)
            for m, t in zip(ev["match_id"], ev["team"], strict=False)
        ]
        ev = ev.dropna(subset=["shots_against"])
        if ev.empty:
            return None

        means, eff = weighted_means(ev, ["shots", "shots_against"], as_of=as_of)
        for_w, against_w = means["shots"], means["shots_against"]
        league_avg = float(ev["shots"].mean())
        team_for = {t: shrink(for_w[t], eff[t], league_avg) for t in for_w}
        team_against = {t: shrink(against_w[t], eff[t], league_avg) for t in against_w}

        totals = ev.groupby("match_id")["shots"].sum()
        mean_t = float(totals.mean())
        dispersion = max(1.0, float(totals.var()) / mean_t) if mean_t > 0 else 1.0
        return cls(team_for=team_for, team_against=team_against,
                   league_avg=league_avg, dispersion=dispersion)

    def _side(self, attacker: str, defender: str) -> float:
        att = self.team_for.get(attacker, self.league_avg)
        deff = self.team_against.get(defender, self.league_avg)
        if self.league_avg <= 0:
            return att
        return att * deff / self.league_avg

    def predict(self, home: str, away: str) -> TotalShotsPrediction:
        h = self._side(home, away)
        a = self._side(away, home)
        return TotalShotsPrediction(home_shots=h, away_shots=a, total=h + a)
