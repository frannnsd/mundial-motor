"""Tests del reporte multi-mercado por partido (Fase 8d)."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.elo_model import EloModel
from mundial_bot.report import build_match_report, format_match_report


def _elo() -> EloModel:
    elo = EloModel()
    elo.ratings.update({"Brazil": 2050, "Morocco": 1950})
    return elo


def _events() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append({"match_id": i, "team": "Brazil", "opponent": "Morocco",
                     "corners_for": 7, "corners_against": 3, "cards": 1, "fouls": 9,
                     "referee": "Ref"})
        rows.append({"match_id": i, "team": "Morocco", "opponent": "Brazil",
                     "corners_for": 3, "corners_against": 7, "cards": 2, "fouls": 13,
                     "referee": "Ref"})
    return pd.DataFrame(rows)


def test_report_sin_modelos_de_goles_solo_da_ganador():
    r = build_match_report("Brazil", "Morocco", elo=_elo())
    assert r.winner.pick == "Brazil"          # favorito por rating
    assert r.winner.fair_odds == pytest.approx(round(1 / r.winner.prob, 2))
    assert r.goals is None and r.corners is None and r.cards is None


def test_report_con_corners_y_cards():
    ev = _events()
    r = build_match_report(
        "Brazil", "Morocco",
        elo=_elo(),
        corners=CornersModel.from_events(ev),
        cards=CardsModel.from_events(ev),
        referee="Ref",
        match_name="Marruecos vs Brasil",
    )

    assert r.corners is not None
    assert "córners" in r.corners.pick
    assert r.corners.expected > 0
    assert r.cards is not None
    assert "tarjetas" in r.cards.pick
    # Cuota justa = 1/prob.
    assert r.corners.fair_odds == pytest.approx(round(1 / r.corners.prob, 2))


def test_format_match_report_arma_texto_legible():
    ev = _events()
    r = build_match_report(
        "Brazil", "Morocco", elo=_elo(),
        corners=CornersModel.from_events(ev),
        cards=CardsModel.from_events(ev),
        referee="Ref", match_name="Marruecos vs Brasil",
    )
    txt = format_match_report(r)
    assert "Marruecos vs Brasil" in txt
    assert "Gana:" in txt
    assert "Córners" in txt
    assert "Tarjetas" in txt
    assert "justo @" in txt
