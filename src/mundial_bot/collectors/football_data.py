"""Colector de cuotas históricas de clubes desde football-data.co.uk (Fase 2: backtest CLV).

Fuente: https://www.football-data.co.uk  (CSV por liga y temporada, con cuotas de
múltiples casas). Nos interesan las cuotas de cierre y apertura de Pinnacle, que son
el "precio justo" de referencia contra el cual medimos Closing Line Value (CLV).

Cubre 3 ligas top (Premier, La Liga, Serie A) por 10 temporadas. Es data de CLUBES,
NO del Mundial: sirve exclusivamente para validar el motor de detección de valor
(¿nuestras señales le ganan al cierre de Pinnacle?) antes de confiar en él en vivo.

Diseño (igual que ``results.py``): la descarga (red) está separada del parseo (puro),
así el parseo se testea con un fixture local sin tocar internet. Cada CSV crudo se
cachea en disco; las descargas respetan un delay entre requests para no abusar del sitio.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from mundial_bot.config import CACHE_DIR

logger = logging.getLogger(__name__)

# Directorio de cache propio de este colector (aislado de otros CSV en data/cache/).
FOOTBALL_DATA_CACHE_DIR = CACHE_DIR / "football_data"

# Ligas soportadas: código football-data.co.uk → nombre completo del torneo.
LEAGUE_NAMES: dict[str, str] = {
    "E0": "England Premier League",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
}
DEFAULT_LEAGUES: tuple[str, ...] = ("E0", "SP1", "I1")

# Temporadas en formato football-data (AAss, ej "1415" = 2014/15). 10 temporadas.
DEFAULT_SEASONS: tuple[str, ...] = (
    "1415", "1516", "1617", "1718", "1819",
    "1920", "2021", "2122", "2223", "2324",
)

# Plantilla de URL del CSV crudo por temporada/división.
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

DOWNLOAD_TIMEOUT_S = 30
DEFAULT_DELAY_S = 1.5

# Columnas base del CSV de origen (siempre presentes en cualquier temporada).
BASE_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")

# Columnas de cuotas Pinnacle: cierre (PSC*) y apertura (PS*). Las temporadas viejas
# pueden NO traer las de cierre → se toleran como ausentes (quedan NaN).
CLOSING_ODDS_COLUMNS = ("PSCH", "PSCD", "PSCA")
OPENING_ODDS_COLUMNS = ("PSH", "PSD", "PSA")


def _cache_path(league: str, season: str) -> Path:
    """Ruta del CSV crudo cacheado para una liga/temporada dada."""
    return FOOTBALL_DATA_CACHE_DIR / f"{league}_{season}.csv"


def download_football_data(
    *,
    leagues: tuple[str, ...] = DEFAULT_LEAGUES,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    force_download: bool = False,
    delay_s: float = DEFAULT_DELAY_S,
) -> list[Path]:
    """Descarga los CSV crudos de football-data.co.uk a la cache local.

    Baja un CSV por cada combinación (liga, temporada) y lo guarda en
    ``data/cache/football_data/{div}_{season}.csv``. Degrada con elegancia:
    un fallo de red en un CSV se loguea como warning y NO frena a los demás.

    Args:
        leagues: códigos de liga football-data (ej. ``("E0", "SP1", "I1")``).
        seasons: temporadas en formato ``AAss`` (ej. ``"2324"`` = 2023/24).
        force_download: si True, re-descarga aunque el CSV ya esté cacheado.
        delay_s: pausa (segundos) entre descargas para respetar el rate limit.

    Returns:
        Rutas de los CSV disponibles en cache (los que ya existían + los bajados
        con éxito). Los que fallaron y no tenían cache previa quedan afuera.
    """
    FOOTBALL_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for league in leagues:
        for season in seasons:
            path = _cache_path(league, season)

            if path.exists() and not force_download:
                paths.append(path)
                continue

            url = URL_TEMPLATE.format(season=season, div=league)
            try:
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S)
                resp.raise_for_status()
            except requests.RequestException as exc:  # noqa: BLE001 — red inestable
                logger.warning("Salteo %s %s (fallo de red): %s", league, season, exc)
                continue

            path.write_bytes(resp.content)
            paths.append(path)
            logger.info("Descargado %s %s → %s", league, season, path.name)

            # Delay entre descargas reales (no tras leer de cache) para no abusar del sitio.
            if delay_s > 0:
                time.sleep(delay_s)

    return paths


def parse_football_csv(csv_text: str, *, league: str, season: str) -> pd.DataFrame:
    """Parsea un CSV crudo de football-data.co.uk al esquema del repo. Función PURA.

    Sin red: recibe el texto del CSV y devuelve el DataFrame normalizado y tipado.
    Descarta filas sin marcador o sin cuota de cierre de Pinnacle (loguea cuántas).

    Args:
        csv_text: contenido crudo del CSV.
        league: código de la liga (ej. ``"E0"``), usado para ``league``/``tournament``.
        season: temporada (ej. ``"2324"``), usada en ``season`` y ``match_id``.

    Returns:
        DataFrame con las columnas normalizadas exactas del esquema (ver módulo).
    """
    df = pd.read_csv(StringIO(csv_text))

    missing_base = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing_base:
        raise ValueError(
            f"Faltan columnas base en {league} {season}: {sorted(missing_base)}"
        )

    n_original = len(df)

    # Fecha: dd/mm/yy o dd/mm/yyyy (varía entre temporadas). format="mixed" + dayfirst
    # parsea cada fila sin el warning de "could not infer format" y sin malinterpretar el día.
    date = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")

    # Cuotas: las columnas ausentes se rellenan con NaN (temporadas viejas sin PSC*).
    def _odds(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("float64")
        # Columna ausente (temporada vieja sin PSC*): NaN float, no pd.NA.
        return pd.Series(float("nan"), index=df.index, dtype="float64")

    tournament = LEAGUE_NAMES.get(league, league)

    out = pd.DataFrame({
        "date": date,
        "home_team": df["HomeTeam"].astype("string"),
        "away_team": df["AwayTeam"].astype("string"),
        "home_score": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_score": pd.to_numeric(df["FTAG"], errors="coerce"),
        "tournament": tournament,
        "neutral": False,
        "league": league,
        "season": season,
        "psc_h": _odds("PSCH"),
        "psc_d": _odds("PSCD"),
        "psc_a": _odds("PSCA"),
        "ps_h": _odds("PSH"),
        "ps_d": _odds("PSD"),
        "ps_a": _odds("PSA"),
    })

    # Descartar filas sin marcador/fecha (partidos suspendidos, filas sucias) y
    # sin cuota de cierre de Pinnacle (sin cierre no hay CLV que medir).
    out = out.dropna(
        subset=["date", "home_score", "away_score", "psc_h", "psc_d", "psc_a"]
    )
    n_kept = len(out)
    n_dropped = n_original - n_kept
    if n_dropped:
        logger.info(
            "%s %s: descarto %d/%d filas (sin marcador o sin cierre Pinnacle)",
            league, season, n_dropped, n_original,
        )

    out = out.reset_index(drop=True)

    # Tipos finales: scores como int, cuotas como float, texto como str de Python.
    # (el astype(int) es seguro porque el dropna de arriba ya sacó los scores NaN;
    # si se relaja ese dropna, hay que castear con Int64 nullable en su lugar.)
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)
    for col in ("psc_h", "psc_d", "psc_a", "ps_h", "ps_d", "ps_a"):
        out[col] = out[col].astype(float)

    # match_id único y estable dentro de (liga, temporada), post-reset de índice.
    out["match_id"] = [f"{league}_{season}_{i}" for i in range(len(out))]

    return out


def load_football_data(
    *,
    leagues: tuple[str, ...] = DEFAULT_LEAGUES,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    force_download: bool = False,
) -> pd.DataFrame:
    """Carga TODA la data histórica de football-data.co.uk como un solo DataFrame.

    Descarga los CSV que falten en cache, parsea cada uno con su liga/temporada,
    concatena todo, ordena por fecha y reinicia el índice.

    Args:
        leagues: códigos de liga a cargar.
        seasons: temporadas a cargar.
        force_download: si True, re-descarga aunque haya cache.

    Returns:
        DataFrame con el esquema normalizado, ordenado por ``date`` ascendente.
        Vacío (con las columnas correctas) si no se pudo cargar ningún CSV.
    """
    download_football_data(
        leagues=leagues, seasons=seasons, force_download=force_download
    )

    frames: list[pd.DataFrame] = []
    for league in leagues:
        for season in seasons:
            path = _cache_path(league, season)
            if not path.exists():
                continue
            try:
                csv_text = path.read_text(encoding="utf-8", errors="replace")
                frames.append(
                    parse_football_csv(csv_text, league=league, season=season)
                )
            except (ValueError, pd.errors.ParserError) as exc:  # noqa: BLE001
                logger.warning("Salteo %s %s (parseo): %s", league, season, exc)
                continue

    if not frames:
        logger.warning("No se cargó ningún CSV de football-data.co.uk")
        return _empty_frame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _empty_frame() -> pd.DataFrame:
    """DataFrame vacío con el esquema normalizado exacto (para degradación elegante)."""
    columns = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral", "league", "season", "match_id",
        "psc_h", "psc_d", "psc_a", "ps_h", "ps_d", "ps_a",
    ]
    return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
