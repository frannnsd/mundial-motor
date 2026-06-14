"""Colector de resultados históricos internacionales (dataset martj42).

Fuente: https://github.com/martj42/international_results  (~50k partidos, 1872→hoy)
Es la base de entrenamiento de todos los modelos: cada partido internacional de
la historia con marcador, torneo y si fue en cancha neutral.

Diseño: la descarga (red) está separada del parseo (puro), así el parseo se
testea con un fixture local sin tocar internet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from mundial_bot.config import CACHE_DIR

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
CACHE_FILE = CACHE_DIR / "international_results.csv"

# Columnas esperadas en el CSV de origen.
EXPECTED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}

DOWNLOAD_TIMEOUT_S = 30


def download_results(*, force: bool = False, timeout: int = DOWNLOAD_TIMEOUT_S) -> Path:
    """Descarga el CSV de resultados a la cache. Devuelve la ruta del archivo.

    Si ya está cacheado y ``force`` es False, no vuelve a bajarlo.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not force:
        return CACHE_FILE

    resp = requests.get(RESULTS_URL, timeout=timeout)
    resp.raise_for_status()
    CACHE_FILE.write_bytes(resp.content)
    return CACHE_FILE


def parse_results(csv_text: str) -> pd.DataFrame:
    """Parsea el CSV crudo a un DataFrame validado y tipado.

    Función pura (sin red): recibe el texto del CSV y devuelve el DataFrame listo.
    - ``date`` → datetime
    - ``neutral`` → bool
    - scores → enteros (descarta partidos sin marcador)
    """
    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset de resultados: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)

    # Descartar filas sin marcador o sin fecha (partidos futuros / datos sucios).
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_results(
    *,
    since: str | None = None,
    exclude_friendlies: bool = False,
    force_download: bool = False,
) -> pd.DataFrame:
    """Carga los resultados históricos como DataFrame, con filtros opcionales.

    Args:
        since: fecha mínima ISO (ej. "2018-01-01") para acotar el histórico.
        exclude_friendlies: si True, descarta amistosos (tournament == "Friendly").
        force_download: si True, re-descarga aunque haya cache.
    """
    path = download_results(force=force_download)
    df = parse_results(path.read_text(encoding="utf-8"))

    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    if exclude_friendlies:
        df = df[df["tournament"].str.lower() != "friendly"]

    return df.reset_index(drop=True)
