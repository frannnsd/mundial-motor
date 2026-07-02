"""Tests del colector de partidos internacionales (nt_data) — SIN red.

Todo corre sobre fixtures sintéticos con la forma exacta de API-Football:
mapeo de match_type, regla de neutral, None→0 con stats presentes, exclusión
de partidos sin stats y esquema de columnas exacto de la tabla consolidada.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mundial_bot.collectors.nt_data import (
    build_nt_match_table,
    classify_match_type,
    fixture_to_row,
    is_neutral,
    load_coverage,
)

# Esquema esperado, HARDCODEADO a propósito: si el módulo cambia una columna,
# este test tiene que romper (los cerebros consumen este contrato).
EXPECTED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "corners_h", "corners_a", "shots_h", "shots_a", "sot_h", "sot_a",
    "fouls_h", "fouls_a", "yellows_h", "yellows_a", "reds_h", "reds_a",
    "ht_goals_h", "ht_goals_a", "league", "season", "match_id",
    "match_type", "neutral", "kickoff_utc",
]

FULL_STATS = {
    "Corner Kicks": 5, "Total Shots": 12, "Shots on Goal": 6,
    "Fouls": 10, "Yellow Cards": 2, "Red Cards": 1,
}


def _stats_block(team_id: int, name: str, values: dict) -> dict:
    return {
        "team": {"id": team_id, "name": name},
        "statistics": [{"type": t, "value": v} for t, v in values.items()],
    }


def _detail(
    *,
    fixture_id: int = 900001,
    league_name: str = "Friendlies",
    home: str = "Argentina",
    away: str = "Chile",
    date: str = "2024-03-26T21:00:00+00:00",
    status: str = "FT",
    goals: tuple[int, int] = (2, 0),
    halftime: tuple[int | None, int | None] = (1, 0),
    stats_home: dict | None = None,
    stats_away: dict | None = None,
    include_stats_key: bool = True,
) -> dict:
    """Fixture sintético con la forma de la respuesta de /fixtures?ids=."""
    detail = {
        "fixture": {"id": fixture_id, "date": date, "status": {"short": status}},
        "league": {"id": 10, "name": league_name, "season": 2024},
        "teams": {"home": {"id": 26, "name": home}, "away": {"id": 27, "name": away}},
        "goals": {"home": goals[0], "away": goals[1]},
        "score": {
            "halftime": {"home": halftime[0], "away": halftime[1]},
            "fulltime": {"home": goals[0], "away": goals[1]},
        },
    }
    if include_stats_key:
        detail["statistics"] = [
            _stats_block(26, home, stats_home if stats_home is not None else FULL_STATS),
            _stats_block(27, away, stats_away if stats_away is not None else FULL_STATS),
        ]
    return detail


# ============================================================================
# match_type
# ============================================================================


@pytest.mark.parametrize(
    ("league_name", "expected"),
    [
        ("World Cup", "mundial"),
        ("World Cup - Qualification South America", "eliminatoria"),
        ("World Cup - Qualification Europe", "eliminatoria"),
        ("UEFA Nations League", "nations_league"),
        ("CONCACAF Nations League", "nations_league"),
        ("Friendlies", "amistoso"),
        ("Copa America", "continental"),
        ("Euro Championship", "continental"),
        ("Africa Cup of Nations", "continental"),
        ("Asian Cup", "continental"),
        ("Gold Cup", "continental"),
        ("Gulf Cup", "continental"),  # contiene 'Cup' y no es mundial/eliminatoria
        ("CONMEBOL - UEFA Finalissima", "otro"),
        ("", "otro"),
    ],
)
def test_classify_match_type(league_name: str, expected: str) -> None:
    assert classify_match_type(league_name) == expected


# ============================================================================
# neutral
# ============================================================================


@pytest.mark.parametrize(
    ("match_type", "home", "season_year", "expected"),
    [
        # Mundial 2026: neutral salvo anfitriones USA/Mexico/Canada.
        ("mundial", "Argentina", 2026, True),
        ("mundial", "USA", 2026, False),
        ("mundial", "Mexico", 2026, False),
        ("mundial", "Canada", 2026, False),
        # Mundial 2022: Qatar anfitrión.
        ("mundial", "Qatar", 2022, False),
        ("mundial", "Brazil", 2022, True),
        # Continental: neutral directo.
        ("continental", "Argentina", 2024, True),
        ("continental", "USA", 2024, True),
        # Localía real en el resto.
        ("eliminatoria", "Argentina", 2025, False),
        ("amistoso", "Argentina", 2024, False),
        ("nations_league", "Spain", 2024, False),
        ("otro", "Italy", 2022, False),
    ],
)
def test_is_neutral(match_type: str, home: str, season_year: int, expected: bool) -> None:
    assert is_neutral(match_type, home, season_year) is expected


# ============================================================================
# fixture_to_row — None→0 con stats presentes / exclusión sin stats
# ============================================================================


def test_none_to_zero_when_stats_block_present() -> None:
    """Con bloque de stats presente, un valor None (ej. Red Cards) es 0 real."""
    stats_home = dict(FULL_STATS, **{"Red Cards": None, "Corner Kicks": None})
    detail = _detail(stats_home=stats_home)
    row = fixture_to_row(detail)
    assert row is not None
    assert row["reds_h"] == 0
    assert row["corners_h"] == 0
    assert row["corners_a"] == FULL_STATS["Corner Kicks"]
    assert row["shots_h"] == FULL_STATS["Total Shots"]


def test_excludes_fixture_without_stats_block() -> None:
    """Sin bloque de statistics el partido se excluye (no se rellena con ceros)."""
    assert fixture_to_row(_detail(include_stats_key=False)) is None


def test_excludes_fixture_with_all_none_stats() -> None:
    """Bloque presente pero TODO None (ej. Finalissima 2022) = sin stats → excluido."""
    empty = {t: None for t in FULL_STATS}
    assert fixture_to_row(_detail(stats_home=empty, stats_away=empty)) is None


def test_excludes_not_finished_and_pre_2022() -> None:
    assert fixture_to_row(_detail(status="NS")) is None
    assert fixture_to_row(_detail(date="2021-11-16T00:00:00+00:00")) is None


def test_finished_includes_aet_and_pen() -> None:
    """La convención del repo de 'terminado' incluye alargue y penales."""
    assert fixture_to_row(_detail(status="AET")) is not None
    assert fixture_to_row(_detail(status="PEN")) is not None


def test_row_fields_and_ht_goals_nan_when_missing() -> None:
    detail = _detail(
        league_name="World Cup",
        home="Jordan",
        away="Argentina",
        date="2026-06-28T02:00:00+00:00",
        goals=(1, 3),
        halftime=(None, None),
    )
    row = fixture_to_row(detail)
    assert row is not None
    assert row["home_score"] == 1 and row["away_score"] == 3
    assert row["ht_goals_h"] is None and row["ht_goals_a"] is None
    assert row["league"] == "NT"
    assert row["season"] == "2026"
    assert row["match_id"] == "900001"
    assert row["match_type"] == "mundial"
    assert row["neutral"] is True  # Jordan no es anfitrión 2026
    assert row["kickoff_utc"] == pd.Timestamp("2026-06-28 02:00:00")
    assert row["date"] == pd.Timestamp("2026-06-28")


# ============================================================================
# build_nt_match_table — esquema exacto, exclusión y cobertura
# ============================================================================


def _write_detail(tmp_path, detail: dict) -> None:
    fid = detail["fixture"]["id"]
    (tmp_path / f"fixture_{fid}.json").write_text(
        json.dumps(detail, ensure_ascii=False), encoding="utf-8"
    )


def test_build_table_exact_schema_and_coverage(tmp_path) -> None:
    _write_detail(tmp_path, _detail(fixture_id=1, league_name="Copa America"))
    _write_detail(
        tmp_path,
        _detail(fixture_id=2, league_name="Copa America", include_stats_key=False),
    )
    _write_detail(tmp_path, _detail(fixture_id=3, league_name="Friendlies",
                                    halftime=(None, None)))

    df = build_nt_match_table(cache_dir=tmp_path, force=True)

    # Esquema EXACTO (orden incluido): contrato con los cerebros.
    assert list(df.columns) == EXPECTED_COLUMNS
    # El partido sin stats quedó afuera de la tabla...
    assert len(df) == 2
    assert set(df["match_id"]) == {"1", "3"}
    # ...pero contado en la cobertura por competencia.
    coverage = load_coverage(cache_dir=tmp_path)
    assert coverage["Copa America"] == {"total": 2, "con_stats": 1, "sin_stats": 1}
    assert coverage["Friendlies"] == {"total": 1, "con_stats": 1, "sin_stats": 0}

    # Tipos clave del contrato.
    assert str(df["date"].dtype).startswith("datetime64")
    assert str(df["kickoff_utc"].dtype).startswith("datetime64")
    assert df["neutral"].dtype == bool
    assert str(df["ht_goals_h"].dtype) == "Int64"
    assert df["ht_goals_h"].isna().sum() == 1  # el amistoso sin halftime → NaN, no 0
    assert (df["league"] == "NT").all()

    # Consolidado en disco y re-lectura cache-primero con el mismo esquema.
    assert (tmp_path / "nt_matches.csv").exists()
    df2 = build_nt_match_table(cache_dir=tmp_path)  # sin force → lee el CSV
    assert list(df2.columns) == EXPECTED_COLUMNS
    assert len(df2) == 2
    assert str(df2["ht_goals_h"].dtype) == "Int64"


def test_build_table_empty_cache(tmp_path) -> None:
    df = build_nt_match_table(cache_dir=tmp_path, force=True)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.empty
