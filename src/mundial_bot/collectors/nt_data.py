"""Colector de partidos INTERNACIONALES (selecciones) desde API-Football — Parte 1 WC-live.

Baja el histórico 2022+ de las selecciones del Mundial 2026 con estadísticas de
equipo (córners, remates, remates al arco, faltas, amarillas, rojas) y lo expone
como un DataFrame con el MISMO esquema que ``football_data.load_football_stats``
— más tres columnas nuevas: ``match_type``, ``neutral`` y ``kickoff_utc`` — para
que los cerebros lo consuman sin cambios.

Diseño (igual que el resto de los colectores): la red está separada del parseo
puro, cache-primero en ``data/nt_cache/`` (gitignored) — un JSON que ya está en
disco NUNCA se re-pide — y presupuesto duro de llamadas con contador expuesto
(``api_calls_made()``).

Endpoints (verificados por sonda):
- ``GET /fixtures?team=<id>&last=99``  → últimos ~99 partidos de una selección.
- ``GET /fixtures?ids=<a-b-c>`` (máx 20 ids) → detalle COMPLETO con ``statistics``.
- ``GET /teams?league=1&season=2026``  → los 48 equipos del Mundial.
- ``GET /fixtures?league=1&season=2026&status=NS`` → fixtures por jugar (vivas).

Aproximaciones documentadas:
- "Terminado" = status ∈ {FT, AET, PEN} (convención del repo, ver
  ``fixtures._FINISHED``). En partidos con alargue el bloque de stats cubre los
  120', pero los goles se toman de ``score.fulltime`` (90') — inconsistencia
  menor, conocida y preferible a perder las finales/llaves decididas en alargue.
- ``neutral``: mundial → True salvo que el local sea anfitrión (2022: Qatar;
  2026: USA/Mexico/Canada); continental → True directo (aprox: sede única);
  eliminatorias/amistosos/nations/otro → False (aprox: localía real).
- Partidos SIN bloque de stats — o con bloque presente pero TODO en None, como
  la Finalissima 2022 — se EXCLUYEN de la tabla (no se rellenan con ceros) y se
  cuentan por competencia en ``coverage.json``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mundial_bot.config import DATA_DIR

logger = logging.getLogger(__name__)

# Cache propio de este colector (gitignored; ver data/nt_cache/ en .gitignore).
NT_CACHE_DIR = DATA_DIR / "nt_cache"

API_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
WC_SEASON = 2026
TIMEOUT_S = 30

# Presupuesto DURO de esta tarea (el plan tiene 7.500/día y el bot vivo necesita quota).
MAX_API_CALLS = 1200
CALL_DELAY_S = 0.4
# Umbral para encarar la Prioridad 2 (las 48 selecciones, no solo las vivas).
PRIORITY2_CALL_THRESHOLD = 800

BATCH_SIZE = 20          # máx ids por llamada a /fixtures?ids=
HISTORY_LAST_N = 99      # /fixtures?team=<id>&last=99 cubre 2022+ para selecciones
MIN_KICKOFF_UTC = pd.Timestamp("2022-01-01")
STOP_CHECK_MIN_MATCHES = 10   # STOP-WARNING si una viva tiene < 10 partidos con stats
STOP_CHECK_MAX_TEAMS = 8      # ... y son 8 o más las vivas en esa situación

# Partido terminado (misma convención que fixtures._FINISHED del repo).
FINISHED_STATUSES = frozenset({"FT", "AET", "PEN"})

# Tipos de stat de API-Football → prefijo de columna normalizada del repo.
STAT_TYPES: dict[str, str] = {
    "Corner Kicks": "corners",
    "Total Shots": "shots",
    "Shots on Goal": "sot",
    "Fouls": "fouls",
    "Yellow Cards": "yellows",
    "Red Cards": "reds",
}

# Torneos continentales conocidos (nombre exacto de API-Football).
CONTINENTAL_EXACT = frozenset({
    "Copa America",
    "Euro Championship",
    "Africa Cup of Nations",
    "Asian Cup",
    "Gold Cup",
    "CONCACAF Gold Cup",
    "AFC Asian Cup",
})

# Anfitriones de Mundial por temporada (para la regla de neutral en 'mundial').
WC_HOSTS_BY_SEASON: dict[int, frozenset[str]] = {
    2022: frozenset({"Qatar"}),
    2026: frozenset({"USA", "Mexico", "Canada"}),
}

# Esquema EXACTO de la tabla (load_football_stats + match_type/neutral/kickoff_utc).
NT_COLUMNS: tuple[str, ...] = (
    "date", "home_team", "away_team", "home_score", "away_score",
    "corners_h", "corners_a", "shots_h", "shots_a", "sot_h", "sot_a",
    "fouls_h", "fouls_a", "yellows_h", "yellows_a", "reds_h", "reds_a",
    "ht_goals_h", "ht_goals_a", "league", "season", "match_id",
    "match_type", "neutral", "kickoff_utc",
)

# ============================================================================
# Presupuesto y llamadas HTTP
# ============================================================================

_api_calls = 0


class BudgetExceededError(RuntimeError):
    """Se alcanzó el presupuesto duro de llamadas de esta tarea."""


def api_calls_made() -> int:
    """Cantidad de llamadas HTTP reales hechas a API-Football en este proceso."""
    return _api_calls


def _get(api_key: str, path: str, params: dict[str, Any]) -> dict:
    """Una llamada HTTP a API-Football: cuenta contra el presupuesto y respeta el delay."""
    global _api_calls
    if not api_key:
        raise RuntimeError("Falta API_FOOTBALL_KEY para consultar API-Football.")
    if _api_calls >= MAX_API_CALLS:
        raise BudgetExceededError(
            f"Presupuesto de {MAX_API_CALLS} llamadas agotado ({_api_calls} hechas)."
        )
    resp = requests.get(
        f"{API_BASE}/{path}",
        params=params,
        headers={"x-apisports-key": api_key},
        timeout=TIMEOUT_S,
    )
    _api_calls += 1
    resp.raise_for_status()
    data = resp.json()
    errors = data.get("errors")
    if errors:  # API-Football devuelve errors={} o [] cuando está todo bien
        raise RuntimeError(f"API-Football error en /{path} {params}: {errors}")
    paging = data.get("paging") or {}
    if (paging.get("total") or 1) > 1:
        logger.warning("/%s %s tiene %s páginas; solo se usa la 1", path, params, paging["total"])
    time.sleep(CALL_DELAY_S)
    return data


def _cached_or_get(api_key: str, cache_path: Path, path: str, params: dict[str, Any]) -> dict:
    """CACHE-PRIMERO: si el JSON está en disco NO se re-llama a la API."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    data = _get(api_key, path, params)
    _save_json(cache_path, data)
    return data


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ============================================================================
# Descarga (red + cache)
# ============================================================================


def wc_team_ids(api_key: str) -> dict[int, str]:
    """Los 48 equipos del Mundial 2026: {team_id: nombre API-Football}."""
    data = _cached_or_get(
        api_key, NT_CACHE_DIR / "wc_teams.json",
        "teams", {"league": WORLD_CUP_LEAGUE_ID, "season": WC_SEASON},
    )
    out: dict[int, str] = {}
    for item in data.get("response", []):
        team = item.get("team") or {}
        if team.get("id") and team.get("name"):
            out[int(team["id"])] = str(team["name"])
    return out


def alive_team_ids(api_key: str) -> dict[int, str]:
    """Selecciones VIVAS: las que aparecen en fixtures NS (por jugar) del Mundial."""
    data = _cached_or_get(
        api_key, NT_CACHE_DIR / "wc_fixtures_ns.json",
        "fixtures", {"league": WORLD_CUP_LEAGUE_ID, "season": WC_SEASON, "status": "NS"},
    )
    out: dict[int, str] = {}
    for item in data.get("response", []):
        teams = item.get("teams") or {}
        for side in ("home", "away"):
            team = teams.get(side) or {}
            if team.get("id") and team.get("name"):
                out[int(team["id"])] = str(team["name"])
    return out


def fetch_team_history(api_key: str, team_id: int) -> list[dict]:
    """Últimos ~99 partidos de una selección, filtrados a TERMINADOS desde 2022.

    Cache: ``nt_cache/team_{id}_last99.json`` (la respuesta cruda completa).
    """
    data = _cached_or_get(
        api_key, NT_CACHE_DIR / f"team_{team_id}_last99.json",
        "fixtures", {"team": team_id, "last": HISTORY_LAST_N},
    )
    out: list[dict] = []
    for item in data.get("response", []):
        if not _is_eligible(item):
            continue
        out.append(item)
    return out


def fetch_fixture_details(api_key: str, fixture_ids: list[int] | set[int]) -> None:
    """Baja el detalle completo (con statistics) de fixtures en batches de 20 ids.

    DEDUPE global + cache POR FIXTURE (``nt_cache/fixture_{id}.json``): un id ya
    cacheado NO se vuelve a pedir. Los ids pedidos que la API no devuelve se
    guardan como stub (``{"_missing": true}``) para no re-pedirlos en cada corrida.
    """
    ids = sorted({int(i) for i in fixture_ids})
    missing = [i for i in ids if not (NT_CACHE_DIR / f"fixture_{i}.json").exists()]
    if not missing:
        return
    n_batches = math.ceil(len(missing) / BATCH_SIZE)
    projected = api_calls_made() + n_batches
    if projected > MAX_API_CALLS:
        raise BudgetExceededError(
            f"Proyección {projected} llamadas (> {MAX_API_CALLS}): "
            f"{len(missing)} fixtures sin cache necesitan {n_batches} batches."
        )
    logger.info("Bajando %d fixtures en %d batches (%d ya en cache)",
                len(missing), n_batches, len(ids) - len(missing))
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        data = _get(api_key, "fixtures", {"ids": "-".join(str(i) for i in batch)})
        seen: set[int] = set()
        for item in data.get("response", []):
            fid = (item.get("fixture") or {}).get("id")
            if not fid:
                continue
            _save_json(NT_CACHE_DIR / f"fixture_{int(fid)}.json", item)
            seen.add(int(fid))
        for fid in set(batch) - seen:
            logger.warning("Fixture %d pedido pero no devuelto por la API; guardo stub", fid)
            _save_json(NT_CACHE_DIR / f"fixture_{fid}.json", {"_missing": True,
                                                              "fixture": {"id": fid}})


# ============================================================================
# Parseo (puro, sin red) — testeable con fixtures sintéticos
# ============================================================================


def classify_match_type(league_name: str) -> str:
    """Mapea league.name de API-Football a match_type del repo.

    ∈ {mundial, continental, eliminatoria, nations_league, amistoso, otro}.
    El orden importa: 'World Cup - Qualification ...' contiene 'Cup' pero es
    eliminatoria, y 'World Cup' exacto es el mundial en sí.
    """
    name = (league_name or "").strip()
    if name == "World Cup":
        return "mundial"
    if "Qualification" in name:
        return "eliminatoria"
    if "Nations League" in name:
        return "nations_league"
    if "Friendlies" in name:
        return "amistoso"
    if name in CONTINENTAL_EXACT or "Cup" in name:
        return "continental"
    return "otro"


def is_neutral(match_type: str, home_team: str, season_year: int) -> bool:
    """Regla de cancha neutral (aproximación honesta, documentada en el módulo).

    - mundial: neutral salvo que el local sea anfitrión de esa edición
      (2022: Qatar; 2026: USA/Mexico/Canada — nombres API-Football tal cual).
    - continental: True directo (casi todos se juegan en sede única).
    - eliminatoria / amistoso / nations_league / otro: False (localía real).
    """
    if match_type == "mundial":
        hosts = WC_HOSTS_BY_SEASON.get(season_year, frozenset())
        return home_team not in hosts
    return match_type == "continental"


def _kickoff_utc(iso_date: str) -> pd.Timestamp | None:
    """Fecha ISO de API-Football → Timestamp UTC naive (o None si no parsea)."""
    ts = pd.to_datetime(iso_date, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.tz_localize(None)


def _is_eligible(item: dict) -> bool:
    """¿Partido TERMINADO (FT/AET/PEN) con kickoff >= 2022-01-01?"""
    fixture = item.get("fixture") or {}
    status = ((fixture.get("status") or {}).get("short") or "").upper()
    if status not in FINISHED_STATUSES:
        return False
    kickoff = _kickoff_utc(fixture.get("date") or "")
    return kickoff is not None and kickoff >= MIN_KICKOFF_UTC


def _stat_int(value: Any) -> int:
    """Valor de stat de API-Football → int. None → 0 (SOLO se llama con bloque presente)."""
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _team_stats(detail: dict) -> tuple[dict[str, int], dict[str, int]] | None:
    """Extrae (stats_local, stats_visitante) del bloque statistics, o None si NO hay stats.

    "No hay stats" = falta el bloque de alguno de los dos equipos, O los bloques
    existen pero TODOS los valores relevantes son None (ej. Finalissima 2022).
    En ese caso el partido se excluye — no se fabrica un 0-0-0-0 falso.
    """
    teams = detail.get("teams") or {}
    home_id = ((teams.get("home") or {}).get("id"))
    away_id = ((teams.get("away") or {}).get("id"))
    if not home_id or not away_id:
        return None

    by_team: dict[int, dict[str, Any]] = {}
    for block in detail.get("statistics") or []:
        tid = ((block.get("team") or {}).get("id"))
        if not tid:
            continue
        by_team[int(tid)] = {
            (it.get("type") or ""): it.get("value")
            for it in (block.get("statistics") or [])
        }
    if int(home_id) not in by_team or int(away_id) not in by_team:
        return None

    raw_h, raw_a = by_team[int(home_id)], by_team[int(away_id)]
    if all(raw.get(t) is None for raw in (raw_h, raw_a) for t in STAT_TYPES):
        return None  # bloque presente pero vacío → partido SIN stats reales

    stats_h = {prefix: _stat_int(raw_h.get(t)) for t, prefix in STAT_TYPES.items()}
    stats_a = {prefix: _stat_int(raw_a.get(t)) for t, prefix in STAT_TYPES.items()}
    return stats_h, stats_a


def fixture_to_row(detail: dict) -> dict[str, Any] | None:
    """Detalle crudo de un fixture → fila normalizada de la tabla NT (función PURA).

    Devuelve None si el partido no es elegible (no terminado / pre-2022 / datos
    incompletos) o si NO tiene estadísticas de equipo (se excluye, no se rellena).
    """
    if not _is_eligible(detail):
        return None
    fixture = detail.get("fixture") or {}
    teams = detail.get("teams") or {}
    home = ((teams.get("home") or {}).get("name"))
    away = ((teams.get("away") or {}).get("name"))
    if not home or not away:
        return None

    kickoff = _kickoff_utc(fixture.get("date") or "")
    if kickoff is None:
        return None

    score = detail.get("score") or {}
    fulltime = score.get("fulltime") or {}
    goals = detail.get("goals") or {}
    # Goles de los 90' (score.fulltime); fallback al marcador final si faltara.
    home_score = fulltime.get("home") if fulltime.get("home") is not None else goals.get("home")
    away_score = fulltime.get("away") if fulltime.get("away") is not None else goals.get("away")
    if home_score is None or away_score is None:
        return None

    stats = _team_stats(detail)
    if stats is None:
        return None
    stats_h, stats_a = stats

    halftime = score.get("halftime") or {}
    ht_h, ht_a = halftime.get("home"), halftime.get("away")

    league_name = ((detail.get("league") or {}).get("name")) or ""
    match_type = classify_match_type(league_name)
    season_year = kickoff.year

    return {
        "date": kickoff.normalize(),
        "home_team": str(home),
        "away_team": str(away),
        "home_score": int(home_score),
        "away_score": int(away_score),
        "corners_h": stats_h["corners"], "corners_a": stats_a["corners"],
        "shots_h": stats_h["shots"], "shots_a": stats_a["shots"],
        "sot_h": stats_h["sot"], "sot_a": stats_a["sot"],
        "fouls_h": stats_h["fouls"], "fouls_a": stats_a["fouls"],
        "yellows_h": stats_h["yellows"], "yellows_a": stats_a["yellows"],
        "reds_h": stats_h["reds"], "reds_a": stats_a["reds"],
        "ht_goals_h": int(ht_h) if ht_h is not None else None,
        "ht_goals_a": int(ht_a) if ht_a is not None else None,
        "league": "NT",
        "season": str(season_year),
        "match_id": str(fixture.get("id")),
        "match_type": match_type,
        "neutral": is_neutral(match_type, str(home), season_year),
        "kickoff_utc": kickoff,
    }


def _frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Filas → DataFrame con el esquema/orden/tipos EXACTOS de NT_COLUMNS (puro)."""
    if not rows:
        return pd.DataFrame({col: pd.Series(dtype="object") for col in NT_COLUMNS})
    df = pd.DataFrame(rows, columns=list(NT_COLUMNS))
    df = df.sort_values(["date", "kickoff_utc", "match_id"]).reset_index(drop=True)
    int_cols = [
        "home_score", "away_score",
        "corners_h", "corners_a", "shots_h", "shots_a", "sot_h", "sot_a",
        "fouls_h", "fouls_a", "yellows_h", "yellows_a", "reds_h", "reds_a",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
    for col in ("ht_goals_h", "ht_goals_a"):
        # halftime puede faltar en partidos viejos → NaN (nullable Int64), NO cero.
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("home_team", "away_team", "league", "season", "match_id", "match_type"):
        df[col] = df[col].astype(str)
    df["neutral"] = df["neutral"].astype(bool)
    df["date"] = pd.to_datetime(df["date"])
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    return df


# ============================================================================
# Tabla consolidada + cobertura
# ============================================================================


def build_nt_match_table(
    *, cache_dir: Path | None = None, force: bool = False
) -> pd.DataFrame:
    """UNA fila por partido internacional con stats — esquema NT_COLUMNS exacto.

    Lee todos los ``fixture_*.json`` del cache, excluye (y CUENTA por competencia
    en ``coverage.json``) los partidos sin stats, y consolida en
    ``nt_cache/nt_matches.csv``. Si el CSV ya existe y no se pide ``force``, se
    lee directo (cache-primero, sin red).
    """
    cache = cache_dir or NT_CACHE_DIR
    csv_path = cache / "nt_matches.csv"
    if csv_path.exists() and not force:
        return _read_table_csv(csv_path)

    rows: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = {}
    for path in sorted(cache.glob("fixture_*.json")):
        try:
            detail = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Salteo %s (JSON inválido): %s", path.name, exc)
            continue
        if not isinstance(detail, dict) or not _is_eligible(detail):
            continue
        league_name = ((detail.get("league") or {}).get("name")) or "?"
        cov = coverage.setdefault(league_name, {"total": 0, "con_stats": 0, "sin_stats": 0})
        cov["total"] += 1
        row = fixture_to_row(detail)
        if row is None:
            cov["sin_stats"] += 1
            continue
        cov["con_stats"] += 1
        rows.append(row)

    df = _frame_from_rows(rows)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not df.empty:
        df.to_csv(csv_path, index=False)
    logger.info("Tabla NT: %d partidos con stats; %d competencias", len(df), len(coverage))
    return df


def _read_table_csv(path: Path) -> pd.DataFrame:
    """Lee el CSV consolidado restaurando los tipos del esquema."""
    df = pd.read_csv(
        path,
        parse_dates=["date", "kickoff_utc"],
        dtype={
            "home_team": str, "away_team": str, "league": str,
            "season": str, "match_id": str, "match_type": str,
        },
    )
    for col in ("ht_goals_h", "ht_goals_a"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["neutral"] = df["neutral"].astype(bool)
    return df[list(NT_COLUMNS)]


def load_coverage(*, cache_dir: Path | None = None) -> dict[str, dict[str, int]]:
    """Cobertura por competencia {league_name: {total, con_stats, sin_stats}}."""
    path = (cache_dir or NT_CACHE_DIR) / "coverage.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def alive_low_coverage(
    df: pd.DataFrame, alive: dict[int, str], *, min_matches: int = STOP_CHECK_MIN_MATCHES
) -> dict[str, int]:
    """Selecciones vivas con menos de ``min_matches`` partidos con stats desde 2022."""
    out: dict[str, int] = {}
    for name in sorted(set(alive.values())):
        n = int(((df["home_team"] == name) | (df["away_team"] == name)).sum()) if len(df) else 0
        if n < min_matches:
            out[name] = n
    return out


# ============================================================================
# Corrida completa (Prioridad 1: vivas → Prioridad 2: las 48)
# ============================================================================


def run_collection(api_key: str) -> dict[str, Any]:
    """Baja histórico + detalles con presupuesto duro y devuelve el resumen.

    Prioridad 1: selecciones vivas (fixtures NS del Mundial). Prioridad 2: el
    resto de las 48, SOLO si el contador va bien (< PRIORITY2_CALL_THRESHOLD).
    Si el presupuesto se agota a mitad de camino, se para y se reporta: el cache
    por fixture hace la corrida 100% resumible.
    """
    summary: dict[str, Any] = {"budget_hit": None, "priority2_done": False}

    wc = wc_team_ids(api_key)
    alive = alive_team_ids(api_key)
    summary["wc_teams"] = len(wc)
    summary["alive_teams"] = len(alive)

    def _history_ids(team_ids: list[int]) -> set[int]:
        ids: set[int] = set()
        for tid in team_ids:
            for item in fetch_team_history(api_key, tid):
                fid = (item.get("fixture") or {}).get("id")
                if fid:
                    ids.add(int(fid))
        return ids

    # --- PRIORIDAD 1: selecciones vivas -------------------------------------
    try:
        p1_ids = _history_ids(sorted(alive))
        fetch_fixture_details(api_key, p1_ids)
        summary["priority1_fixtures"] = len(p1_ids)
    except BudgetExceededError as exc:
        summary["budget_hit"] = f"P1: {exc}"
        logger.error("Presupuesto agotado en Prioridad 1: %s", exc)

    # --- PRIORIDAD 2: el resto de las 48 ------------------------------------
    if summary["budget_hit"] is None and api_calls_made() < PRIORITY2_CALL_THRESHOLD:
        rest = sorted(tid for tid in wc if tid not in alive)
        try:
            p2_ids = _history_ids(rest)
            fetch_fixture_details(api_key, p2_ids)
            summary["priority2_done"] = True
            summary["priority2_teams"] = len(rest)
        except BudgetExceededError as exc:
            summary["budget_hit"] = f"P2: {exc}"
            logger.error("Presupuesto agotado en Prioridad 2: %s", exc)

    # --- Consolidación (no usa red) ------------------------------------------
    df = build_nt_match_table(force=True)
    summary["api_calls"] = api_calls_made()
    summary["table_rows"] = len(df)
    summary["by_match_type"] = (
        df["match_type"].value_counts().to_dict() if len(df) else {}
    )
    summary["coverage"] = load_coverage()
    low = alive_low_coverage(df, alive)
    summary["alive_low_coverage"] = low
    summary["stop_warning"] = len(low) >= STOP_CHECK_MAX_TEAMS
    return summary


def main() -> None:  # pragma: no cover — orquestación con red real
    """CLI: ``python -m mundial_bot.collectors.nt_data`` (usa .env)."""
    import os

    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    summary = run_collection(api_key)

    print("\n===== RESUMEN NT-DATA =====")
    print(f"Llamadas API usadas:   {summary['api_calls']} / {MAX_API_CALLS}")
    print(f"Equipos WC / vivos:    {summary['wc_teams']} / {summary['alive_teams']}")
    print(f"Prioridad 2 completa:  {summary['priority2_done']}")
    if summary["budget_hit"]:
        print(f"PRESUPUESTO AGOTADO:   {summary['budget_hit']}")
    print(f"Filas en la tabla:     {summary['table_rows']}")
    print("Partidos por match_type:")
    for mt, n in sorted(summary["by_match_type"].items(), key=lambda kv: -kv[1]):
        print(f"  {mt:15s} {n}")
    print("Cobertura por competencia (total / con_stats / sin_stats):")
    cov = summary["coverage"]
    for name in sorted(cov, key=lambda k: -cov[k]["total"]):
        c = cov[name]
        print(f"  {name[:52]:52s} {c['total']:4d} / {c['con_stats']:4d} / {c['sin_stats']:4d}")
    low = summary["alive_low_coverage"]
    if low:
        tag = "STOP-WARNING" if summary["stop_warning"] else "aviso"
        print(f"[{tag}] vivas con <{STOP_CHECK_MIN_MATCHES} partidos con stats: {low}")
    else:
        print("Todas las selecciones vivas tienen >= "
              f"{STOP_CHECK_MIN_MATCHES} partidos con stats desde 2022.")


if __name__ == "__main__":  # pragma: no cover
    main()
