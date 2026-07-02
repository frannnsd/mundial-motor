"""Guard anti-leakage point-in-time (Fase 1).

Verifica tres cosas:
1. El guard FALLA si alguna feature usa datos con fecha >= kickoff.
2. `as_of=kickoff` en los modelos de conteo excluye el futuro (== prefiltrar a mano).
3. `as_of=None` reproduce EXACTAMENTE el comportamiento histórico (path live intacto).
"""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.backtest.leakage_guard import (
    LeakageError,
    assert_point_in_time,
    max_feature_date,
)
from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.count_market import weighted_means
from mundial_bot.models.goals_model import GoalsModel
from mundial_bot.models.shots_model import ShotsModel
from mundial_bot.models.total_shots_model import TotalShotsModel

KICKOFF = pd.Timestamp("2026-04-01")  # cae entre el 3er y el 4to partido


def _match_rows(
    mid, date, home, away, cf_h, cf_a, *, cards_h=2, cards_a=2, sot_h=4, sot_a=3, ref="R"
):
    """Dos filas (una por equipo) de un partido, con todas las columnas que usan los modelos."""
    base = {"match_id": mid, "date": pd.Timestamp(date), "referee": ref, "fouls": 10}
    return [
        {**base, "team": home, "opponent": away, "is_home": 1,
         "corners_for": cf_h, "corners_against": cf_a, "cards": cards_h,
         "shots": cf_h * 3, "sot_for": sot_h, "sot_against": sot_a},
        {**base, "team": away, "opponent": home, "is_home": 0,
         "corners_for": cf_a, "corners_against": cf_h, "cards": cards_a,
         "shots": cf_a * 3, "sot_for": sot_a, "sot_against": sot_h},
    ]


def _events(future_cf: float = 99.0) -> pd.DataFrame:
    """3 partidos pre-kickoff + 1 partido FUTURO (>= kickoff) con un valor extremo para A."""
    rows = []
    rows += _match_rows(1, "2026-01-01", "A", "B", 5, 4, cards_h=2, sot_h=4)
    rows += _match_rows(2, "2026-02-01", "A", "B", 6, 3, cards_h=3, sot_h=5)
    rows += _match_rows(3, "2026-03-01", "A", "B", 5, 5, cards_h=2, sot_h=4)
    rows += _match_rows(4, "2026-05-01", "A", "B", future_cf, 4, cards_h=20, sot_h=20)
    return pd.DataFrame(rows)


def _pre_kickoff(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["date"] < KICKOFF].copy()


# --- 1. El guard detecta leakage ---

def test_guard_raises_on_future_row():
    with pytest.raises(LeakageError):
        assert_point_in_time(_events(), KICKOFF, label="córners")


def test_guard_passes_when_point_in_time():
    assert_point_in_time(_pre_kickoff(_events()), KICKOFF, label="córners")  # no levanta


def test_guard_noop_when_as_of_none():
    assert_point_in_time(_events(), None)  # sin kickoff, no chequea nada


def test_max_feature_date():
    assert max_feature_date(_pre_kickoff(_events())) == pd.Timestamp("2026-03-01")
    assert max_feature_date(pd.DataFrame()) is None


# --- 2. as_of excluye el futuro (== prefiltrar) ---

def test_weighted_means_as_of_equals_prefilter():
    ev = _events()
    full = weighted_means(ev, ["corners_for"])[0]["corners_for"]
    pit = weighted_means(ev, ["corners_for"], as_of=KICKOFF)[0]["corners_for"]
    manual = weighted_means(_pre_kickoff(ev), ["corners_for"], as_of=KICKOFF)[0]["corners_for"]
    assert pit["A"] == pytest.approx(manual["A"])          # as_of == prefiltrar
    assert pit["A"] != pytest.approx(full["A"])            # y el futuro SÍ cambiaba el número


def test_corners_as_of_excludes_future():
    full = CornersModel.from_events(_events())
    pit = CornersModel.from_events(_events(), as_of=KICKOFF)
    # Mismo kickoff, con el futuro ya ausente: debe dar idéntico → el futuro se excluye.
    manual = CornersModel.from_events(_pre_kickoff(_events()), as_of=KICKOFF)
    assert pit.team_for["A"] == pytest.approx(manual.team_for["A"])
    assert pit.team_for["A"] != pytest.approx(full.team_for["A"])
    assert pit.predict("A", "B").total != pytest.approx(full.predict("A", "B").total)


def test_cards_as_of_excludes_future():
    full = CardsModel.from_events(_events())
    pit = CardsModel.from_events(_events(), as_of=KICKOFF)
    manual = CardsModel.from_events(_pre_kickoff(_events()), as_of=KICKOFF)
    assert pit.team_cards["A"] == pytest.approx(manual.team_cards["A"])
    assert pit.team_cards["A"] != pytest.approx(full.team_cards["A"])


# --- 3. as_of=None reproduce EXACTO el comportamiento histórico (path live intacto) ---

@pytest.mark.parametrize("model_cls", [CornersModel, CardsModel, ShotsModel, TotalShotsModel])
def test_as_of_none_is_noop(model_cls):
    ev = _events()  # incluye el partido "futuro": en modo live (as_of=None) no se filtra nada
    default = model_cls.from_events(ev)
    explicit_none = model_cls.from_events(ev, as_of=None)
    assert default is not None and explicit_none is not None
    # Son dataclasses: la igualdad compara TODOS los campos → prueba que None es no-op exacto.
    assert default == explicit_none


# --- 4. goals.fit_calibration respeta el cutoff as_of ---

def _goals_df() -> pd.DataFrame:
    """Partidos con total bajo antes del kickoff y total ALTO después (para que el
    cutoff cambie el factor). Fechas reales para poder filtrar por as_of."""
    rows = []
    for i in range(8):  # pre-kickoff: 2 goles por partido
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1})
    for i in range(8):  # post-kickoff: 6 goles por partido (no deben contar con as_of)
        rows.append({"date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=i),
                     "home_team": "A", "away_team": "B", "home_score": 3, "away_score": 3})
    return pd.DataFrame(rows)


def _stub_goals_model() -> GoalsModel:
    """GoalsModel con score_matrix determinístico (pred=2 goles) — testea la lógica de
    calibración + el cutoff as_of SIN depender de penaltyblog."""
    m = GoalsModel()
    m._teams = {"A", "B"}
    m.score_matrix = lambda h, a, *, neutral=True: (None, 1.0, 1.0)  # type: ignore[method-assign]
    return m


def test_fit_calibration_as_of_ignores_future():
    m_all = _stub_goals_model()
    f_all = m_all.fit_calibration(_goals_df(), min_matches=4, bounds=(0.5, 2.0))

    m_pit = _stub_goals_model()
    f_pit = m_pit.fit_calibration(_goals_df(), min_matches=4, bounds=(0.5, 2.0), as_of=KICKOFF)

    # Con todos los partidos, el factor sube (mete los 6-goles del futuro);
    # con as_of=kickoff, solo cuentan los 2-goles previos → factor ~1.0 y MENOR.
    assert f_pit == pytest.approx(1.0, abs=1e-6)
    assert f_pit < f_all
