"""Test de integración del pipeline (offline, modelos y partidos inyectados)."""

from __future__ import annotations

from pathlib import Path

from mundial_bot.config import Settings
from mundial_bot.models.elo_model import EloModel
from mundial_bot.pipeline import Models, run_pipeline, value_picks_for_match
from mundial_bot.value.odds import load_sample

SAMPLE = Path(__file__).parent / "data" / "sample_odds.json"


def _models() -> Models:
    elo = EloModel()
    # Ratings que crean value claro vs las cuotas del sample.
    elo.ratings.update({
        "Argentina": 2100, "Mexico": 1700,   # Argentina favorito fuerte
        "Spain": 2000, "Brazil": 2060,
    })
    return Models(elo=elo, goals=None)


def _settings(**overrides) -> Settings:
    base = {"bankroll_usd": 100.0, "min_edge": 0.05, "kelly_fraction": 0.25}
    base.update(overrides)
    return Settings().model_copy(update=base)


def test_value_picks_for_match_detecta_value():
    matches = load_sample(SAMPLE)
    picks = value_picks_for_match(matches[0], _models(), min_edge=0.05)

    # Argentina @2.00 con prob de modelo ~0.9 → value enorme.
    assert any(p.selection.selection == "home" and p.edge > 0.3 for p in picks)
    # Cada value pick trae la prob justa del mercado (de-vig).
    assert all(p.fair_prob is not None for p in picks)


def test_run_pipeline_genera_cartilla_completa():
    matches = load_sample(SAMPLE)

    result = run_pipeline(
        settings=_settings(),
        models=_models(),
        matches=matches,
        date_str="14/06/2026",
    )

    assert result.n_matches == 2
    assert len(result.value_picks) >= 1
    assert "CARTILLA MUNDIAL — 14/06/2026" in result.message
    assert "Argentina" in result.message


def test_run_pipeline_respeta_tope_de_exposicion():
    matches = load_sample(SAMPLE)
    settings = _settings(max_total_exposure_pct=0.25)

    result = run_pipeline(settings=settings, models=_models(), matches=matches,
                          date_str="14/06/2026")

    total_stake = sum(s.stake for s in result.staked)
    # No puede superar el 25% de la banca ($25), con margen de redondeo.
    assert total_stake <= 25.0 + 0.05


def test_run_pipeline_sin_value_no_rompe():
    # Modelo "plano" (todos iguales) → poco o ningún value, pero no debe romper.
    elo = EloModel()
    elo.ratings.update({"Argentina": 1800, "Mexico": 1800, "Spain": 1800, "Brazil": 1800})
    result = run_pipeline(
        settings=_settings(min_edge=0.50),  # umbral altísimo → nada pasa
        models=Models(elo=elo),
        matches=load_sample(SAMPLE),
        date_str="14/06/2026",
    )
    assert "no hay apuestas simples" in result.message.lower()
