"""Router FastAPI de MLB (prefix /mlb) — el contrato que consume la web.

NO se monta acá: el orquestador hace `app.include_router(mlb_api.router)` en
api/app.py. MISMA auth que /wc (header X-Access-Key, importada de wc_api) y
mismas convenciones de payload (claves JSON string; líneas O/U como "8.5").

Los datos salen de Supabase vía wc/store.py con sport='mlb'; el forward-test
usa el MISMO cálculo que el Mundial (jobs.compute_forward_test filtrado).
"""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from mundial_bot.api import wc_api
from mundial_bot.api.wc_api import require_access_key
from mundial_bot.wc import jobs, mlb_jobs, store

AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
SPORT = "mlb"

# Los mercados destacados de la card del día: (id, etiqueta, camino en payload.markets).
_TOP_MARKET_DEFS = (
    ("mlb_total_ou_8.5", "Más de 8.5 carreras", ("totales", "8.5", "over")),
    ("mlb_f5_ou_4.5", "Más de 4.5 carreras F5", ("f5", "totales", "4.5", "over")),
    ("mlb_rl_home_1.5", "Local −1.5 (run line)", ("run_line", "home_-1.5")),
)

router = APIRouter(prefix="/mlb", dependencies=[Depends(require_access_key)])


def _game_card(rep: dict) -> dict:
    """Resumen por card para la home MLB del día (derivado del daily_report)."""
    payload = rep.get("payload") or {}
    markets = payload.get("markets") or {}
    means = payload.get("means") or {}
    top_markets = []
    for market, label, path in _TOP_MARKET_DEFS:
        prob = wc_api._dig(markets, path)  # noqa: SLF001 — reuso deliberado
        if prob is not None:
            top_markets.append({"market": market, "label": label,
                                "prob": round(float(prob), 4)})
    return {
        "game_pk": rep.get("fixture_id"),
        "kickoff_utc": rep.get("kickoff_utc"),
        "home": rep.get("home"),
        "away": rep.get("away"),
        "venue": payload.get("venue"),
        "starters": payload.get("starters"),
        "moneyline": markets.get("moneyline"),
        "top_markets": top_markets,
        "means_compact": {k: round(float(v), 2) for k, v in means.items()
                          if isinstance(v, int | float)},
    }


@router.get("/today")
def today(date: str | None = None) -> dict:
    """Cards de los juegos MLB del día (default: HOY en hora argentina)."""
    wc_api._require_store()  # noqa: SLF001 — reuso deliberado
    if date is not None:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(status_code=422,
                                detail=f"Fecha inválida: {date!r} (usar YYYY-MM-DD).") from exc
    date_str = date or datetime.now(AR_TZ).strftime("%Y-%m-%d")
    reports = store.get_reports(date_str, sport=SPORT)
    return {"date": date_str, "matches": [_game_card(r) for r in reports]}


@router.get("/match/{game_pk}")
def match_detail(game_pk: int) -> dict:
    """Reporte completo del juego (payload con pmfs) + predicciones con cuotas."""
    wc_api._require_store()  # noqa: SLF001
    rep = store.get_report(game_pk, sport=SPORT)
    if rep is None:
        raise HTTPException(status_code=404,
                            detail=f"Sin reporte diario MLB para el juego {game_pk}.")
    predictions = store.select("props_log", {
        "fixture_id": f"eq.{game_pk}", "sport": f"eq.{SPORT}", "order": "id.asc",
    })
    return {"report": rep, "predictions": predictions}


@router.get("/live-odds/{game_pk}")
def live_odds(game_pk: int) -> dict:
    """Cuotas Bet365 actuales del juego (odds-api.io), cacheadas 10 min.

    Solo lectura para que el humano compare y decida — acá no se apuesta nada.
    """
    wc_api._require_store()  # noqa: SLF001
    rep = store.get_report(game_pk, sport=SPORT)
    if rep is None:
        raise HTTPException(status_code=404,
                            detail=f"Sin reporte diario MLB para el juego {game_pk}.")
    return mlb_jobs.get_mlb_odds(game_pk, str(rep["home"]), str(rep["away"]),
                                 str(rep.get("report_date") or "")[:10])


class OddsBody(BaseModel):
    """Carga manual de la línea/cuota bet365 sobre una predicción MLB vigente."""

    fixture_id: int  # game_pk
    player_id: int = 0
    market: str = Field(min_length=1)
    line: float | None = None
    odds: float = Field(gt=1.0)
    stake: float | None = Field(default=None, gt=0)


@router.post("/odds")
def attach_odds(body: OddsBody) -> dict:
    wc_api._require_store()  # noqa: SLF001
    updated = store.ft_attach_odds(
        body.fixture_id, body.player_id, body.market,
        line=body.line, odds=body.odds, stake=body.stake, sport=SPORT,
    )
    out: dict = {"updated": updated}
    if updated == 0:
        out["detail"] = "No había predicción MLB registrada con esa clave (game/player/market)."
    return out


@router.get("/forward-test")
def forward_test() -> dict:
    """Resumen vivo del forward-test MLB (mismo cálculo que el del Mundial)."""
    wc_api._require_store()  # noqa: SLF001
    rows = store.select("props_log", {"sport": f"eq.{SPORT}", "order": "id.asc"})
    return jobs.compute_forward_test(rows, sport=SPORT)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

_RUNNABLE = {
    "mlb_daily": mlb_jobs.run_mlb_daily,
    "mlb_settle": mlb_jobs.run_mlb_settle,
}


@router.post("/admin/run/{job}")
def admin_run(job: str) -> dict:
    """Dispara un job MLB en un thread (verificación e2e y operación manual)."""
    fn = _RUNNABLE.get(job)
    if fn is None:
        raise HTTPException(status_code=404,
                            detail=f"Job desconocido: {job!r} (mlb_daily|mlb_settle).")
    threading.Thread(target=fn, name=f"mlb-job-{job}", daemon=True).start()
    return {"started": True, "job": job}
