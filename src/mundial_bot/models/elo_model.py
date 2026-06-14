"""Modelo Elo internacional (metodología eloratings.net).

Por qué propio y no el Elo de penaltyblog: el Elo de selecciones gana su edge de
dos cosas que la versión genérica no tiene:
  1. K variable por importancia del torneo (una final del Mundial mueve más el
     rating que un amistoso).
  2. Multiplicador por diferencia de gol (un 5-0 mueve más que un 1-0).

La investigación mostró que un Elo así afinado (~60% acierto, RPS 0.171) iguala o
supera a Dixon-Coles y a modelos de ML en datos de selecciones (poca data). Por eso
es la BASE del sistema; Dixon-Coles se reserva para mercados de goles.

Conversión a 1X2: el Elo da un "expected score" (que cuenta el empate como 0.5).
Lo partimos en local/empate/visitante con un modelo de empate parametrizado y
**calibrable** (los parámetros draw_max/draw_width se ajustan a los datos en la
Fase 5; los defaults vienen de la literatura).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# --- K-factors por importancia del torneo (eloratings.net) ---
K_WORLD_CUP_FINALS = 60.0
K_CONTINENTAL_FINALS = 50.0
K_QUALIFIERS_AND_MAJOR = 40.0
K_OTHER = 30.0
K_FRIENDLY = 20.0

# Torneos cuya fase final pesa 50 (campeonatos continentales / intercontinentales).
_CONTINENTAL_FINALS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "afc asian cup",
    "gold cup",
    "confederations cup",
    "nations league",
)


def tournament_k(tournament: str) -> float:
    """Devuelve el K-factor según la importancia del torneo."""
    t = (tournament or "").lower()
    if "friendly" in t:
        return K_FRIENDLY
    if "qualification" in t or "qualifier" in t:
        return K_QUALIFIERS_AND_MAJOR
    if "fifa world cup" in t:
        return K_WORLD_CUP_FINALS
    if any(name in t for name in _CONTINENTAL_FINALS):
        return K_CONTINENTAL_FINALS
    return K_OTHER


def goal_diff_multiplier(margin: int) -> float:
    """Multiplicador por diferencia de gol (eloratings.net).

    1 para empate o diferencia de 1; 1.5 para diferencia de 2; (11+N)/8 para 3+.
    """
    n = abs(int(margin))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8.0


@dataclass
class MatchProbabilities:
    """Probabilidades 1X2 de un partido."""

    home: float
    draw: float
    away: float

    def as_dict(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}


@dataclass
class EloConfig:
    base_rating: float = 1500.0
    home_advantage: float = 100.0
    # Modelo de empate (calibrable en Fase 5).
    draw_max: float = 0.29
    draw_width: float = 300.0


@dataclass
class EloModel:
    """Sistema de ratings Elo + predicción de probabilidades 1X2."""

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    # --- Ratings ---
    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.base_rating)

    def _rating_diff(self, home: str, away: str, *, neutral: bool) -> float:
        hfa = 0.0 if neutral else self.config.home_advantage
        return self.rating(home) - self.rating(away) + hfa

    def expected_score(self, home: str, away: str, *, neutral: bool = False) -> float:
        """Expectativa de puntaje del local en [0,1] (empate cuenta 0.5)."""
        dr = self._rating_diff(home, away, neutral=neutral)
        return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    # --- Predicción 1X2 ---
    def predict(self, home: str, away: str, *, neutral: bool = False) -> MatchProbabilities:
        """Convierte el expected score en probabilidades local/empate/visitante.

        Garantiza que sumen 1 y que el expected score se conserve:
        E = P(local) + 0.5·P(empate).
        """
        we = self.expected_score(home, away, neutral=neutral)
        dr = self._rating_diff(home, away, neutral=neutral)

        # Empate decrece (gaussiana) a medida que crece la diferencia de rating.
        p_draw = self.config.draw_max * math.exp(-((dr / self.config.draw_width) ** 2))
        p_home = we - p_draw / 2.0
        p_away = (1.0 - we) - p_draw / 2.0

        # Clamp de seguridad ante diferencias extremas y renormalización.
        p_home = max(p_home, 1e-6)
        p_away = max(p_away, 1e-6)
        total = p_home + p_draw + p_away
        return MatchProbabilities(p_home / total, p_draw / total, p_away / total)

    # --- Actualización tras un partido ---
    def update(
        self,
        home: str,
        away: str,
        *,
        home_score: int,
        away_score: int,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> None:
        """Actualiza los ratings tras un resultado (suma cero entre ambos)."""
        we = self.expected_score(home, away, neutral=neutral)
        if home_score > away_score:
            w = 1.0
        elif home_score < away_score:
            w = 0.0
        else:
            w = 0.5

        k = tournament_k(tournament)
        g = goal_diff_multiplier(home_score - away_score)
        delta = k * g * (w - we)

        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    # --- Entrenamiento sobre histórico ---
    def fit(self, df: pd.DataFrame) -> EloModel:
        """Recorre los partidos en orden cronológico y actualiza ratings.

        Espera columnas: date, home_team, away_team, home_score, away_score,
        tournament, neutral.
        """
        ordered = df.sort_values("date")
        for row in ordered.itertuples(index=False):
            self.update(
                row.home_team,
                row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                tournament=getattr(row, "tournament", "Friendly"),
                neutral=bool(getattr(row, "neutral", False)),
            )
        return self

    def rankings(self, top: int | None = None) -> list[tuple[str, float]]:
        """Ranking de equipos por rating (de mayor a menor)."""
        ordered = sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:top] if top else ordered
