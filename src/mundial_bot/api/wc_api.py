"""Router FastAPI del Mundial (prefix /wc) — el contrato que consume la web.

NO se monta acá: el orquestador hace `app.include_router(wc_api.router)` en
api/app.py. Todas las rutas exigen el header `X-Access-Key` == env WEB_ACCESS_KEY
(503 si la env no está seteada; 401 si el header falta o no coincide).

Los datos salen de Supabase vía wc/store.py (daily_reports + props_log); el
cálculo del forward-test es el MISMO que usa el job weekly (jobs.compute_forward_test).
Convención del payload: claves JSON siempre string (las líneas O/U son "2.5", "9.5"...).
"""

from __future__ import annotations

import json
import hmac
import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from mundial_bot.wc import jobs, store

AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# Los 3-4 mercados destacados de la card del día: (id, etiqueta, camino en markets90).
_TOP_MARKET_DEFS = (
    ("goles_ou_2.5", "Más de 2.5 goles", ("goles_ou", "2.5", "over")),
    ("corners_ou_9.5", "Más de 9.5 córners", ("corners_ou", "9.5", "over")),
    ("tarjetas_ou_3.5", "Más de 3.5 amarillas", ("tarjetas_ou", "3.5", "over")),
    ("btts", "Ambos anotan", ("btts", "yes")),
)


def require_access_key(x_access_key: str | None = Header(default=None)) -> None:
    """Auth de TODAS las rutas /wc: header X-Access-Key == env WEB_ACCESS_KEY."""
    expected = os.environ.get("WEB_ACCESS_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="WEB_ACCESS_KEY no está configurada en el backend: "
                   "setearla en el entorno para habilitar la API /wc.",
        )
    if not hmac.compare_digest((x_access_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="X-Access-Key inválida o ausente.")


router = APIRouter(prefix="/wc", dependencies=[Depends(require_access_key)])


def _require_store() -> None:
    if not store.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase no configurado (faltan SUPABASE_URL/SUPABASE_SERVICE_KEY).",
        )


def _dig(d: dict, path: tuple[str, ...]) -> float | None:
    cur: object = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, int | float) else None


def _match_card(rep: dict) -> dict:
    """Resumen por card para la home del día (derivado del daily_report)."""
    payload = rep.get("payload") or {}
    m90 = payload.get("markets90") or {}
    ko = payload.get("knockout") or {}
    means = payload.get("means") or {}
    top_markets = []
    for market, label, path in _TOP_MARKET_DEFS:
        prob = _dig(m90, path)
        if prob is not None:
            top_markets.append({"market": market, "label": label,
                                "prob": round(float(prob), 4)})
    return {
        "fixture_id": rep.get("fixture_id"),
        "kickoff_utc": rep.get("kickoff_utc"),
        "home": rep.get("home"),
        "away": rep.get("away"),
        "round": rep.get("round"),
        "is_knockout": bool(rep.get("is_knockout")),
        "xi_confirmed": bool(rep.get("xi_confirmed")),
        "one_x_two": m90.get("1x2"),
        "se_clasifica": ko.get("se_clasifica"),
        "top_markets": top_markets,
        "means_compact": {k: round(float(v), 2) for k, v in means.items()
                          if isinstance(v, int | float)},
    }


@router.get("/today")
def today(date: str | None = None) -> dict:
    """Cards de los partidos del día (default: HOY en hora argentina)."""
    _require_store()
    if date is not None:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(status_code=422,
                                detail=f"Fecha inválida: {date!r} (usar YYYY-MM-DD).") from exc
    date_str = date or datetime.now(AR_TZ).strftime("%Y-%m-%d")
    reports = store.get_reports(date_str)
    return {"date": date_str, "matches": [_match_card(r) for r in reports]}


@router.get("/match/{fixture_id}")
def match_detail(fixture_id: int) -> dict:
    """Reporte completo del partido (payload con pmfs) + predicciones con cuotas."""
    _require_store()
    rep = store.get_report(fixture_id)
    if rep is None:
        raise HTTPException(status_code=404,
                            detail=f"Sin reporte diario para el fixture {fixture_id}.")
    predictions = store.select("props_log", {"fixture_id": f"eq.{fixture_id}",
                                             "order": "id.asc"})
    return {"report": rep, "predictions": predictions}


class OddsBody(BaseModel):
    """Carga manual de la línea/cuota bet365 sobre una predicción vigente."""

    fixture_id: int
    player_id: int = 0
    market: str = Field(min_length=1)
    line: float | None = None
    odds: float = Field(gt=1.0)
    stake: float | None = Field(default=None, gt=0)


@router.post("/odds")
def attach_odds(body: OddsBody) -> dict:
    _require_store()
    updated = store.ft_attach_odds(
        body.fixture_id, body.player_id, body.market,
        line=body.line, odds=body.odds, stake=body.stake,
    )
    out: dict = {"updated": updated}
    if updated == 0:
        out["detail"] = "No había predicción registrada con esa clave (fixture/player/market)."
    return out


@router.get("/forward-test")
def forward_test() -> dict:
    """Resumen vivo del forward-test (mismo cálculo que el job weekly)."""
    _require_store()
    rows = store.select("props_log", {"order": "id.asc"})
    return jobs.compute_forward_test(rows)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

_RUNNABLE = {
    "daily": jobs.run_daily,
    "lineups": jobs.run_lineups,
    "settle": jobs.run_settle,
    "weekly": jobs.run_weekly,
}


@router.get("/admin/status")
def admin_status() -> dict:
    """Últimos job_runs + estado del scheduler + quota de API-Football del proceso."""
    from mundial_bot.collectors import nt_data, players_wc
    from mundial_bot.wc import scheduler as wc_scheduler

    job_runs: list[dict] = []
    if store.is_configured():
        try:
            job_runs = store.latest_job_runs()
        except requests.RequestException as exc:
            job_runs = [{"error": f"No pude leer job_runs: {exc}"}]
    nt_calls = nt_data.api_calls_made()
    pl_calls = players_wc.api_calls_made()
    return {
        "store_configured": store.is_configured(),
        "jobs": job_runs,
        "scheduler": wc_scheduler.scheduler_status(),
        "quota_hoy": {"nt_data": nt_calls, "players": pl_calls,
                      "total": nt_calls + pl_calls},
    }


@router.post("/admin/run/{job}")
def admin_run(job: str) -> dict:
    """Dispara un job en un thread (para la verificación e2e y la operación manual)."""
    fn = _RUNNABLE.get(job)
    if fn is None:
        raise HTTPException(status_code=404,
                            detail=f"Job desconocido: {job!r} (daily|lineups|settle|weekly).")
    threading.Thread(target=fn, name=f"wc-job-{job}", daemon=True).start()
    return {"started": True, "job": job}


@router.get("/admin/backup")
def admin_backup() -> Response:
    """Dump completo del forward-test (props_log) como JSON descargable."""
    _require_store()
    rows = store.select("props_log", {"order": "id.asc"})
    filename = f"props_log_{datetime.now(AR_TZ).strftime('%Y%m%d')}.json"
    return Response(
        content=json.dumps(rows, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
