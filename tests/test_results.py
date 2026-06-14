"""Tests del colector de resultados históricos (Agente 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from mundial_bot.collectors import results as R

# Fixture: CSV mínimo con el formato real del dataset martj42.
SAMPLE_CSV = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,True\n"
    "2022-12-14,Argentina,Croatia,3,0,FIFA World Cup,Lusail,Qatar,True\n"
    "2026-07-19,Spain,Brazil,,,FIFA World Cup,New York,USA,True\n"  # futuro, sin score
    "2021-06-05,Argentina,Brazil,1,0,Friendly,Buenos Aires,Argentina,False\n"
)


def test_parse_results_tipa_y_ordena_por_fecha():
    # Act
    df = R.parse_results(SAMPLE_CSV)

    # Assert: tipos correctos y orden ascendente por fecha.
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_bool_dtype(df["neutral"])
    assert list(df["date"]) == sorted(df["date"])


def test_parse_results_descarta_partidos_sin_marcador():
    # Act
    df = R.parse_results(SAMPLE_CSV)

    # Assert: la fila futura Spain-Brazil (sin score) se descarta.
    assert len(df) == 3
    assert "Spain" not in set(df["home_team"])
    assert df["home_score"].dtype == int


def test_parse_results_falla_si_faltan_columnas():
    # Arrange
    bad_csv = "date,home_team\n2022-01-01,Argentina\n"

    # Act / Assert
    with pytest.raises(ValueError, match="Faltan columnas"):
        R.parse_results(bad_csv)


def test_load_results_filtra_amistosos(monkeypatch, tmp_path):
    # Arrange: evitamos la red escribiendo el CSV en la cache y cortando la descarga.
    cache_file = tmp_path / "international_results.csv"
    cache_file.write_text(SAMPLE_CSV, encoding="utf-8")
    monkeypatch.setattr(R, "CACHE_FILE", cache_file)
    monkeypatch.setattr(R, "download_results", lambda **_: cache_file)

    # Act
    df = R.load_results(exclude_friendlies=True)

    # Assert: el amistoso Argentina-Brazil queda afuera.
    assert (df["tournament"].str.lower() != "friendly").all()
    assert len(df) == 2


def test_load_results_filtra_por_fecha(monkeypatch, tmp_path):
    # Arrange
    cache_file = tmp_path / "international_results.csv"
    cache_file.write_text(SAMPLE_CSV, encoding="utf-8")
    monkeypatch.setattr(R, "download_results", lambda **_: cache_file)

    # Act
    df = R.load_results(since="2022-01-01")

    # Assert: solo quedan los dos partidos de 2022.
    assert len(df) == 2
    assert (df["date"] >= pd.Timestamp("2022-01-01")).all()


@pytest.mark.network
def test_download_results_real():
    """Descarga real del dataset (se saltea sin red con: pytest -m 'not network')."""
    # Act
    df = R.load_results(since="2024-01-01", force_download=True)

    # Assert: el dataset real tiene miles de partidos y las columnas clave.
    assert len(df) > 100
    assert {"home_team", "away_team", "home_score"}.issubset(df.columns)
