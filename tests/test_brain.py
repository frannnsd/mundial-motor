"""Tests del cerebro conversable (parseo de equipos + respuestas)."""

from __future__ import annotations

from mundial_bot.brain import BotBrain, parse_two_teams, resolve_team
from mundial_bot.models.elo_model import EloModel
from mundial_bot.pipeline import Models

KNOWN = {"Argentina", "Brazil", "Mexico", "Croatia", "United States", "Morocco"}


def test_resolve_team_ingles_espanol_y_fuzzy():
    assert resolve_team("Argentina", KNOWN) == "Argentina"
    assert resolve_team("brasil", KNOWN) == "Brazil"        # español
    assert resolve_team("méxico", KNOWN) == "Mexico"         # acento
    assert resolve_team("marruecos", KNOWN) == "Morocco"     # español
    assert resolve_team("xyz123", KNOWN) is None


def test_parse_two_teams_varios_separadores():
    assert parse_two_teams("Argentina vs México", KNOWN) == ("Argentina", "Mexico")
    assert parse_two_teams("Brasil - Croacia", KNOWN) == ("Brazil", "Croatia")
    assert parse_two_teams("Mexico x Brazil", KNOWN) == ("Mexico", "Brazil")
    assert parse_two_teams("hola que tal", KNOWN) is None


def _brain() -> BotBrain:
    elo = EloModel()
    elo.ratings.update({"Argentina": 2100, "Brazil": 2050})
    return BotBrain(models=Models(elo=elo, goals=None), corners=None, cards=None)


def test_handle_text_predice_partido_valido():
    txt = _brain().handle_text("Argentina vs Brasil")
    assert "Argentina" in txt
    assert "Gana" in txt
    assert "%" in txt


def test_handle_text_no_reconoce_da_ayuda():
    txt = _brain().handle_text("contame un chiste")
    assert "No reconocí" in txt
