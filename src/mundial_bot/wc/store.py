"""Storage compartido motor ↔ web: Supabase Postgres vía PostgREST (solo requests).

Sin dependencias nuevas: Supabase expone la base por HTTP (PostgREST); este módulo
es el ÚNICO punto de acceso. La web NUNCA toca Supabase directo — consume la API
del backend, que usa esto.

Config por entorno (no se piden por chat ni se hardcodean):
  SUPABASE_URL          ej. https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY  service_role key (SOLO backend; jamás en el frontend)

Sin las env vars, `is_configured()` es False y el pipeline local sigue usando
SQLite/CSV como siempre (dev). Semántica clave (idéntica al forward-test local):
las predicciones son INMUTABLES — inserción con on_conflict=ignore (la primera
gana); las cuotas/liquidación solo completan campos de esa fila.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 25


def _base() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/")


def _key() -> str:
    return os.environ.get("SUPABASE_SERVICE_KEY", "")


def is_configured() -> bool:
    return bool(_base() and _key())


def _headers(*, prefer: str | None = None) -> dict[str, str]:
    h = {
        "apikey": _key(),
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _url(table: str) -> str:
    return f"{_base()}/rest/v1/{table}"


def select(table: str, params: dict[str, str] | None = None) -> list[dict]:
    """SELECT vía PostgREST. params = filtros PostgREST (ej. {"report_date": "eq.2026-07-02"})."""
    r = requests.get(_url(table), headers=_headers(), params=params or {}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def insert_ignore(table: str, rows: list[dict], on_conflict: str) -> int:
    """INSERT ... ON CONFLICT DO NOTHING (la primera predicción gana). Devuelve insertadas."""
    if not rows:
        return 0
    r = requests.post(
        _url(table) + f"?on_conflict={on_conflict}",
        headers=_headers(prefer="resolution=ignore-duplicates,return=representation"),
        data=json.dumps(rows),
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return len(r.json())


def upsert(table: str, rows: list[dict], on_conflict: str) -> None:
    """UPSERT (para payloads diarios / tablas de datos, donde la última versión vale)."""
    if not rows:
        return
    r = requests.post(
        _url(table) + f"?on_conflict={on_conflict}",
        headers=_headers(prefer="resolution=merge-duplicates"),
        data=json.dumps(rows),
        timeout=_TIMEOUT,
    )
    r.raise_for_status()


def update(table: str, filters: dict[str, str], patch: dict) -> int:
    """UPDATE con filtros PostgREST. Devuelve filas afectadas."""
    r = requests.patch(
        _url(table), headers=_headers(prefer="return=representation"),
        params=filters, data=json.dumps(patch), timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return len(r.json())


# ---------------------------------------------------------------------------
# Dominio: forward-test (misma semántica que forward_test/log.py, en Postgres)
# ---------------------------------------------------------------------------

def ft_log_prediction(**kw: Any) -> bool:
    """Registra una predicción (inmutable: si ya existe, NO se pisa)."""
    row = {k: v for k, v in kw.items() if v is not None}
    row.setdefault("player_id", 0)
    row.setdefault("player_name", "-")
    n = insert_ignore("props_log", [row], on_conflict="fixture_id,player_id,market")
    return n > 0


def ft_attach_odds(
    fixture_id: int, player_id: int, market: str,
    *, line: float | None, odds: float, stake: float | None = None,
) -> int:
    """Adjunta la cuota de bet365 a la predicción VIGENTE (snapshot inmutable: los
    campos pred_* de la fila no se tocan — el EV se calcula contra lo que el modelo
    decía al momento de la carga)."""
    patch: dict[str, Any] = {
        "odds": odds, "book": "bet365",
        "odds_added_at": datetime.now(UTC).isoformat(),
    }
    if line is not None:
        patch["line"] = line
    if stake is not None:
        patch["stake"] = stake
    return update("props_log", {
        "fixture_id": f"eq.{fixture_id}",
        "player_id": f"eq.{player_id}",
        "market": f"eq.{market}",
    }, patch)


def ft_settle_rows(rows: list[dict]) -> int:
    """Aplica liquidaciones {id, actual, brier} (solo filas aún no liquidadas)."""
    n = 0
    now = datetime.now(UTC).isoformat()
    for r in rows:
        n += update("props_log", {
            "id": f"eq.{r['id']}", "settled_at": "is.null",
        }, {"actual": r["actual"], "brier": r.get("brier"), "settled_at": now})
    return n


def ft_pending(fixture_id: int) -> list[dict]:
    return select("props_log", {
        "fixture_id": f"eq.{fixture_id}", "settled_at": "is.null",
    })


# ---------------------------------------------------------------------------
# Dominio: payloads diarios, datos, jobs, backup
# ---------------------------------------------------------------------------

def save_daily_report(row: dict) -> None:
    upsert("daily_reports", [row], on_conflict="fixture_id")


def get_reports(date_str: str) -> list[dict]:
    return select("daily_reports", {
        "report_date": f"eq.{date_str}", "order": "kickoff_utc.asc",
    })


def get_report(fixture_id: int) -> dict | None:
    rows = select("daily_reports", {"fixture_id": f"eq.{fixture_id}"})
    return rows[0] if rows else None


def job_start(job: str) -> int | None:
    r = requests.post(
        _url("job_runs"), headers=_headers(prefer="return=representation"),
        data=json.dumps({"job": job, "started_at": datetime.now(UTC).isoformat()}),
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    body = r.json()
    return body[0]["id"] if body else None


def job_finish(run_id: int | None, *, status: str, detail: str = "", api_calls: int = 0) -> None:
    if run_id is None:
        return
    try:
        update("job_runs", {"id": f"eq.{run_id}"}, {
            "finished_at": datetime.now(UTC).isoformat(),
            "status": status, "detail": detail[:2000], "api_calls": api_calls,
        })
    except requests.RequestException:  # la observabilidad nunca tumba el job
        logger.warning("No pude cerrar el job_run %s", run_id)


def latest_job_runs() -> list[dict]:
    return select("job_runs", {"order": "started_at.desc", "limit": "20"})


def daily_backup() -> int:
    """Dump del forward-test completo a la tabla backups (1/día, idempotente)."""
    rows = select("props_log", {"order": "id.asc"})
    today = datetime.now(UTC).date().isoformat()
    upsert("backups", [{"backup_date": today, "payload": rows}], on_conflict="backup_date")
    return len(rows)
