"""Tests del colector de football-data.co.uk (Fase 2: backtest CLV).

Todos sin red: el parseo es puro y se prueba con un fixture inline que imita el
formato real del CSV. La única prueba que toca internet está marcada @network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.collectors import football_data as FD

# Fixture: CSV con columnas reales de football-data.co.uk. Incluye:
#   - fila con fecha dd/mm/yy (formato viejo)
#   - filas normales con cierre Pinnacle (PSC*) y apertura (PS*)
#   - una fila SIN cierre (PSC* vacío) → debe descartarse
#   - una fila SIN marcador (FTHG/FTAG vacío) → debe descartarse
SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,PSCH,PSCD,PSCA\n"
    "E0,17/08/19,Liverpool,Norwich,4,1,H,1.25,6.50,13.0,1.22,6.80,15.0\n"
    "E0,18/08/2019,West Ham,Man City,0,5,A,8.00,5.20,1.40,8.50,5.30,1.38\n"
    "E0,24/08/2019,Arsenal,Tottenham,2,2,D,2.10,3.60,3.50,2.05,3.70,3.60\n"
    "E0,25/08/2019,Chelsea,Leicester,1,1,D,1.70,4.00,5.00,,,\n"  # sin cierre → descarte
    "E0,31/08/2019,Everton,Wolves,,,,2.00,3.40,4.00,1.95,3.50,4.10\n"  # sin marcador
)

# Fixture de temporada vieja: SIN columnas de cierre PSC* (deben quedar NaN, no crashear).
OLD_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA\n"
    "SP1,30/08/14,Sevilla,Valencia,1,1,D,2.40,3.20,3.00\n"
)


def test_parse_columnas_normalizadas_exactas():
    # Act
    df = FD.parse_football_csv(SAMPLE_CSV, league="E0", season="1920")

    # Assert: esquema exacto que consume el backtest.
    expected = {
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral", "league", "season", "match_id",
        "psc_h", "psc_d", "psc_a", "ps_h", "ps_d", "ps_a",
    }
    assert set(df.columns) == expected


def test_parse_descarta_sin_marcador_y_sin_cierre():
    # Act
    df = FD.parse_football_csv(SAMPLE_CSV, league="E0", season="1920")

    # Assert: de 5 filas quedan 3 (se van Chelsea sin cierre y Everton sin marcador).
    assert len(df) == 3
    homes = set(df["home_team"])
    assert "Chelsea" not in homes  # descartada por falta de cierre
    assert "Everton" not in homes  # descartada por falta de marcador


def test_parse_fecha_dayfirst():
    # Act
    df = FD.parse_football_csv(SAMPLE_CSV, league="E0", season="1920")

    # Assert: la fila dd/mm/yy (17/08/19) se parsea como 2019-08-17, no 2017-08-19.
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    liverpool = df[df["home_team"] == "Liverpool"].iloc[0]
    assert liverpool["date"] == pd.Timestamp("2019-08-17")


def test_parse_tipos_int_y_float():
    # Act
    df = FD.parse_football_csv(SAMPLE_CSV, league="E0", season="1920")

    # Assert: scores enteros, cuotas float.
    assert df["home_score"].dtype == int
    assert df["away_score"].dtype == int
    for col in ("psc_h", "psc_d", "psc_a", "ps_h", "ps_d", "ps_a"):
        assert df[col].dtype == float
    # Los valores de cierre se leen bien.
    liverpool = df[df["home_team"] == "Liverpool"].iloc[0]
    assert liverpool["psc_h"] == pytest.approx(1.22)


def test_parse_metadatos_liga_y_match_id():
    # Act
    df = FD.parse_football_csv(SAMPLE_CSV, league="E0", season="1920")

    # Assert: liga, torneo, neutral y match_id únicos y con el prefijo esperado.
    assert (df["league"] == "E0").all()
    assert (df["tournament"] == "England Premier League").all()
    assert (df["neutral"] == False).all()  # noqa: E712 — comparación vectorizada
    assert df["match_id"].is_unique
    assert df["match_id"].iloc[0] == "E0_1920_0"


def test_parse_temporada_vieja_sin_cierre_no_crashea():
    # Act: CSV sin columnas PSC* → todas las filas se descartan (no hay cierre),
    # pero NO debe crashear y las columnas de cierre existen como float (NaN).
    df = FD.parse_football_csv(OLD_CSV, league="SP1", season="1415")

    # Assert: esquema completo presente aunque no queden filas.
    assert "psc_h" in df.columns
    assert df["psc_h"].dtype == float
    assert len(df) == 0  # sin cierre Pinnacle → descarte total


def test_parse_falla_si_faltan_columnas_base():
    # Arrange: falta HomeTeam/AwayTeam/scores.
    bad_csv = "Div,Date\nE0,17/08/19\n"

    # Act / Assert
    with pytest.raises(ValueError, match="Faltan columnas base"):
        FD.parse_football_csv(bad_csv, league="E0", season="1920")


def test_load_concatena_ordena_y_resetea(monkeypatch, tmp_path):
    # Arrange: escribimos dos CSV en una cache falsa y cortamos la descarga.
    cache = tmp_path / "football_data"
    cache.mkdir()
    (cache / "E0_1920.csv").write_text(SAMPLE_CSV, encoding="utf-8")
    (cache / "SP1_1415.csv").write_text(OLD_CSV, encoding="utf-8")
    monkeypatch.setattr(FD, "FOOTBALL_DATA_CACHE_DIR", cache)
    monkeypatch.setattr(FD, "_cache_path", lambda league, season: cache / f"{league}_{season}.csv")
    monkeypatch.setattr(FD, "download_football_data", lambda **_: [])

    # Act
    df = FD.load_football_data(leagues=("E0", "SP1"), seasons=("1920", "1415"))

    # Assert: solo entran las 3 filas válidas de E0 (SP1 vieja se descarta), ordenadas.
    assert len(df) == 3
    assert list(df["date"]) == sorted(df["date"])
    assert list(df.index) == [0, 1, 2]


@pytest.mark.network
def test_download_real_una_temporada(tmp_path, monkeypatch):
    """Descarga real de un CSV (se saltea sin red con: pytest -m 'not network')."""
    # Arrange: cache aislada para no pisar la real.
    cache = tmp_path / "football_data"
    monkeypatch.setattr(FD, "FOOTBALL_DATA_CACHE_DIR", cache)
    monkeypatch.setattr(FD, "_cache_path", lambda league, season: cache / f"{league}_{season}.csv")

    # Act
    df = FD.load_football_data(leagues=("E0",), seasons=("2324",), force_download=True)

    # Assert: una temporada de Premier tiene cientos de partidos con cierre.
    assert len(df) > 100
    assert {"psc_h", "psc_d", "psc_a", "match_id"}.issubset(df.columns)
