"""Colector de estadísticas de córners, tarjetas y faltas desde StatsBomb open-data.

martj42 solo trae goles. Para predecir córners y tarjetas necesitamos esos eventos:
StatsBomb los publica gratis para torneos internacionales modernos (Mundiales
2018/2022, Euros, Copa América, AFCON).

De cada partido derivamos, por equipo:
  - córners a favor (pass_type == "Corner")
  - tarjetas (foul_committed_card / bad_behaviour_card)
  - faltas cometidas (type == "Foul Committed")
y el árbitro (de la metadata del partido), para modelar su severidad.

La descarga es lenta (un archivo de eventos por partido), así que se cachea el
resultado agregado a CSV: se baja una sola vez.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from mundial_bot.config import CACHE_DIR

warnings.filterwarnings("ignore")

# (competition_id, season_id) de torneos masculinos internacionales modernos.
MENS_TOURNAMENTS: list[tuple[int, int]] = [
    (43, 106),    # FIFA World Cup 2022
    (43, 3),      # FIFA World Cup 2018
    (55, 282),    # UEFA Euro 2024
    (55, 43),     # UEFA Euro 2020
    (223, 282),   # Copa América 2024
    (1267, 107),  # African Cup of Nations 2023
]

EVENTS_CACHE = CACHE_DIR / "setpiece_events.csv"


def _counts_by_team(events: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Cuenta córners, tarjetas y faltas por equipo en un partido."""
    out: dict[str, dict[str, int]] = {}

    if "pass_type" in events:
        for team, n in events[events["pass_type"] == "Corner"].groupby("team").size().items():
            out.setdefault(team, {})["corners"] = int(n)

    if "type" in events:
        fouls = events[events["type"] == "Foul Committed"].groupby("team").size()
        for team, n in fouls.items():
            out.setdefault(team, {})["fouls"] = int(n)

    card_mask = pd.Series(False, index=events.index)
    for col in ("foul_committed_card", "bad_behaviour_card"):
        if col in events:
            card_mask = card_mask | events[col].notna()
    for team, n in events[card_mask].groupby("team").size().items():
        out.setdefault(team, {})["cards"] = int(n)

    return out


def collect_events(tournaments: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Baja y agrega los eventos de los torneos. Una fila por equipo por partido.

    Columnas: match_id, team, opponent, corners_for, corners_against, cards, fouls, referee.
    """
    from statsbombpy import sb

    tournaments = tournaments or MENS_TOURNAMENTS
    rows: list[dict] = []

    for comp_id, season_id in tournaments:
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception:
            continue
        for mt in matches.itertuples(index=False):
            try:
                events = sb.events(int(mt.match_id))
            except Exception:
                continue
            counts = _counts_by_team(events)
            home, away = mt.home_team, mt.away_team
            referee = getattr(mt, "referee", None)
            for team, opp in ((home, away), (away, home)):
                c = counts.get(team, {})
                opp_c = counts.get(opp, {})
                rows.append({
                    "match_id": int(mt.match_id),
                    "team": team,
                    "opponent": opp,
                    "corners_for": c.get("corners", 0),
                    "corners_against": opp_c.get("corners", 0),
                    "cards": c.get("cards", 0),
                    "fouls": c.get("fouls", 0),
                    "referee": referee,
                })
    return pd.DataFrame(rows)


def build_cache(tournaments: list[tuple[int, int]] | None = None) -> Path:
    """Baja los eventos y cachea el DataFrame agregado a CSV. Devuelve la ruta."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_events(tournaments)
    df.to_csv(EVENTS_CACHE, index=False, encoding="utf-8")
    return EVENTS_CACHE


def load_events(*, build_if_missing: bool = True) -> pd.DataFrame:
    """Carga los eventos agregados desde la cache (los baja si faltan)."""
    if not EVENTS_CACHE.exists():
        if not build_if_missing:
            raise FileNotFoundError("Cache de StatsBomb ausente. Corré build_cache() primero.")
        build_cache()
    return pd.read_csv(EVENTS_CACHE, encoding="utf-8")
