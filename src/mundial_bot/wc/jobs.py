"""Jobs de automatización del Mundial: funciones puras llamables por el scheduler Y la API.

Cuatro jobs sobre el storage compartido (wc/store.py → Supabase):

  run_daily    Reporte completo por partido NS del día (payload para la web) +
               registro de las mismas ~15-25 predicciones que el pre-day local.
  run_lineups  En la ventana [-60min, kickoff]: pide lineups (máx 1 intento/5min
               por partido, cache-primero), recalcula props con XI confirmado,
               guarda deltas y marca xi_confirmed.
  run_settle   Liquida props y mercados de equipo de los partidos FT/AET/PEN del
               día contra los resultados reales, sube las filas nuevas a
               nt_matches/player_matches y corre el backup diario.
  run_weekly   Resumen del forward-test (mismo cálculo que GET /wc/forward-test)
               + backup; el detalle queda en job_runs.

Observabilidad: cada job queda registrado en job_runs (job_start/job_finish con
status ok/error, detail y el diff de llamadas reales a API-Football). Ante una
excepción se reintenta UNA vez. Si Supabase no está configurado, el job loguea
un warning y sale sin crashear (dev local sigue con SQLite/CSV vía wc/daily.py).

REUSO, no duplicación: la matemática vive en wc/daily.py (helpers privados),
markets/projection.py y forward_test/log.py — acá solo se orquesta contra la nube.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from mundial_bot.collectors import nt_data, players_wc
from mundial_bot.config import get_settings
from mundial_bot.forward_test import log as ft_log
from mundial_bot.markets import projection as proj
from mundial_bot.wc import daily, store

logger = logging.getLogger(__name__)

LINEUP_RETRY_S = 300          # máx 1 intento de lineups cada 5 min por partido
TOP_PROPS_LOGGED = 5          # jugadores por equipo que van al forward-test
TOP_DELTAS = 5                # deltas por equipo al confirmarse el XI
MIN_SETTLED_FOR_WORST = 10    # n liquidadas mínimas para entrar a worst_markets
_DETAIL_MAX = 1900            # el store trunca a 2000; margen propio

# Columnas de props que viajan en el payload de la web (las que existan).
_PROP_COLS = (
    "player_id", "player_name", "position", "exp_minutes",
    "mu_shots", "mu_sot", "mu_goals", "mu_yellow",
    "p_scores", "p_card", "p_shots_2plus", "p_sot_1plus",
)

# Memoria del proceso: último intento de lineups por fixture (rate-limit propio).
_last_lineup_try: dict[int, float] = {}


# ---------------------------------------------------------------------------
# Helpers comunes
# ---------------------------------------------------------------------------

def _api_calls_total() -> int:
    """Llamadas HTTP reales a API-Football en este proceso (ambos colectores)."""
    return nt_data.api_calls_made() + players_wc.api_calls_made()


def _jsonable(value: Any, *, ndigits: int | None = None) -> Any:
    """Convierte a tipos JSON-safe (numpy → py, Timestamp → ISO, claves → str)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v, ndigits=ndigits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v, ndigits=ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v, ndigits=ndigits) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, ndigits) if ndigits is not None else value
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    return value


def _props_records(df: pd.DataFrame) -> list[dict]:
    """Props DataFrame → records JSON-safe con las columnas del contrato web."""
    if df.empty:
        return []
    cols = [c for c in _PROP_COLS if c in df.columns]
    return _jsonable(df[cols].to_dict("records"), ndigits=4)


def _run_job(name: str, fn: Callable[[], dict]) -> dict:
    """Envuelve un job: job_start/job_finish, 1 reintento, api_calls del período."""
    if not store.is_configured():
        logger.warning("Job %s salteado: Supabase no configurado (SUPABASE_URL/KEY).", name)
        return {"status": "skipped", "detail": "store no configurado"}
    calls0 = _api_calls_total()
    run_id = None
    try:
        run_id = store.job_start(name)
    except requests.RequestException as exc:  # la observabilidad no tumba el job
        logger.warning("No pude registrar job_start de %s: %s", name, exc)
    logger.info("Job %s: inicio.", name)
    try:
        try:
            result = fn()
        except Exception as exc:
            logger.warning("Job %s falló (%s); reintento una vez.", name, exc)
            result = fn()
    except Exception as exc:
        logger.exception("Job %s: error definitivo.", name)
        store.job_finish(run_id, status="error",
                         detail=f"{type(exc).__name__}: {exc}"[:_DETAIL_MAX],
                         api_calls=_api_calls_total() - calls0)
        return {"status": "error", "detail": str(exc)}
    detail = json.dumps(result, ensure_ascii=False, default=str)[:_DETAIL_MAX]
    store.job_finish(run_id, status="ok", detail=detail,
                     api_calls=_api_calls_total() - calls0)
    logger.info("Job %s: ok — %s", name, detail)
    return {"status": "ok", **result}


# ---------------------------------------------------------------------------
# daily — reporte + predicciones del día (espejo cloud de cmd_pre_day)
# ---------------------------------------------------------------------------

def run_daily(date_str: str | None = None) -> dict:
    """Payload web + predicciones de cada partido NS del día en Supabase."""
    return _run_job("daily", lambda: _daily(date_str))


def _daily(date_str: str | None) -> dict:
    key = get_settings().api_football_key
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    fixtures = [f for f in daily._day_fixtures(key, date_str)  # noqa: SLF001
                if f["fixture"]["status"]["short"] == "NS"]
    if not fixtures:
        return {"date": date_str, "matches": 0, "predictions": 0}
    engine = daily._build_engine()  # noqa: SLF001
    table = daily._player_table()  # noqa: SLF001

    total_logged = 0
    for fx in fixtures:
        home, away = fx["teams"]["home"]["name"], fx["teams"]["away"]["name"]
        when = pd.Timestamp(fx["fixture"]["date"][:10])
        ko_match = daily._is_knockout(fx)  # noqa: SLF001
        pred = engine.predict_match(home, away, when=when)
        pmfs, means = pred["pmfs"], pred["means"]
        markets = proj.project_all(pmfs)
        ko = proj.knockout_markets(pmfs) if ko_match else None
        p_et = ko["p_prorroga"] if ko else 0.0
        te_factor = 1.0 + p_et * proj.ET_FRACTION * proj.ET_FATIGUE if ko else 1.0
        horizon = "120" if ko_match else "90"
        props_h = daily._props_for(table, home, means, "h",  # noqa: SLF001
                                   horizon=horizon, te_factor=te_factor)
        props_a = daily._props_for(table, away, means, "a",  # noqa: SLF001
                                   horizon=horizon, te_factor=te_factor)

        store.save_daily_report({
            "fixture_id": int(fx["fixture"]["id"]),
            "report_date": date_str,
            "kickoff_utc": fx["fixture"]["date"],
            "home": home,
            "away": away,
            "round": (fx.get("league") or {}).get("round", ""),
            "is_knockout": ko_match,
            "payload": {
                "means": _jsonable(means, ndigits=6),
                "pmfs": {q: _jsonable(p, ndigits=6) for q, p in pmfs.items()},
                "markets90": _jsonable(markets, ndigits=6),
                "knockout": _jsonable(ko, ndigits=6) if ko else None,
                "props": {"home": _props_records(props_h), "away": _props_records(props_a)},
            },
            "xi_confirmed": False,
        })
        total_logged += _log_match_predictions(fx, pred, markets, ko, props_h, props_a)
    return {"date": date_str, "matches": len(fixtures), "predictions": total_logged}


def _log_match_predictions(fx: dict, pred: dict, markets: dict, ko: dict | None,
                           props_h: pd.DataFrame, props_a: pd.DataFrame) -> int:
    """Las MISMAS ~15-25 predicciones que daily._log_match_predictions, en la nube."""
    fid = int(fx["fixture"]["id"])
    match = f"{fx['teams']['home']['name']} vs {fx['teams']['away']['name']}"
    now = datetime.now(UTC).isoformat(timespec="seconds")
    n = 0

    def team(market: str, prob: float | None, mean: float | None = None,
             line: float | None = None) -> None:
        nonlocal n
        n += int(store.ft_log_prediction(
            fixture_id=fid, match=match, market=market, player_id=0, player_name="-",
            pred_mean=mean, pred_prob=prob, line=line, notes=f"as_of={now}",
        ))

    for side in ("home", "draw", "away"):
        team(f"team_1x2_{side}", markets["1x2"][side])
    team("team_btts", markets["btts"]["yes"])
    means = pred.get("means", {})
    team("team_goals_ou_2.5", markets["goles_ou"][2.5]["over"],
         means.get("goals_h", 0) + means.get("goals_a", 0), 2.5)
    team("team_corners_ou_9.5", markets["corners_ou"][9.5]["over"], line=9.5)
    team("team_yellows_ou_3.5", markets["tarjetas_ou"][3.5]["over"], line=3.5)
    if ko is not None:
        for side in ("home", "away"):
            team(f"team_se_clasifica_{side}", ko["se_clasifica"][side])

    for props in (props_h, props_a):
        for _, r in props.head(TOP_PROPS_LOGGED).iterrows():
            n += int(store.ft_log_prediction(
                fixture_id=fid, match=match, market="shots",
                player_id=int(r["player_id"]), player_name=str(r["player_name"]),
                pred_mean=float(r.get("mu_shots", 0)), notes=f"as_of={now}",
            ))
            if r.get("p_scores") is not None:
                n += int(store.ft_log_prediction(
                    fixture_id=fid, match=match, market="anota",
                    player_id=int(r["player_id"]), player_name=str(r["player_name"]),
                    pred_prob=float(r["p_scores"]), notes=f"as_of={now}",
                ))
    return n


# ---------------------------------------------------------------------------
# lineups — XI confirmado en la ventana [-60min, kickoff]
# ---------------------------------------------------------------------------

def run_lineups() -> dict:
    """Confirma XI de los partidos en ventana (barato: sin partidos, sin llamadas)."""
    if not store.is_configured():
        logger.warning("Job lineups salteado: Supabase no configurado.")
        return {"status": "skipped", "detail": "store no configurado"}
    now = pd.Timestamp.now(tz="UTC")
    try:
        candidates = _reports_in_window(now)
    except requests.RequestException as exc:
        logger.warning("lineups: no pude leer daily_reports: %s", exc)
        return {"status": "error", "detail": str(exc)}
    if not candidates:
        return {"status": "ok", "in_window": 0, "attempts": 0, "confirmed": 0}
    return _run_job("lineups", lambda: _lineups(candidates))


def _reports_in_window(now: pd.Timestamp) -> list[dict]:
    """daily_reports sin XI confirmado con kickoff dentro de [now, now+60min]."""
    horizon = now + pd.Timedelta(minutes=daily.LINEUP_WINDOW_MIN)
    dates = sorted({now.strftime("%Y-%m-%d"), horizon.strftime("%Y-%m-%d")})
    out: list[dict] = []
    for d in dates:
        for rep in store.get_reports(d):
            if rep.get("xi_confirmed"):
                continue
            kickoff = pd.Timestamp(rep["kickoff_utc"])
            if kickoff.tzinfo is None:
                kickoff = kickoff.tz_localize("UTC")
            mins = (kickoff - now).total_seconds() / 60.0
            if 0 <= mins <= daily.LINEUP_WINDOW_MIN:
                out.append(rep)
    return out


def _lineups(candidates: list[dict]) -> dict:
    key = get_settings().api_football_key
    engine = None
    table = None
    attempts = confirmed = 0
    for rep in candidates:
        fid = int(rep["fixture_id"])
        if time.monotonic() - _last_lineup_try.get(fid, -1e9) < LINEUP_RETRY_S:
            continue
        _last_lineup_try[fid] = time.monotonic()
        attempts += 1
        lu = daily._get_cached(key, "/fixtures/lineups",  # noqa: SLF001
                               {"fixture": fid}, f"lineups_{fid}")
        teams_lu = lu.get("response", [])
        if len(teams_lu) < 2:
            logger.info("lineups: fixture %d todavía sin XI publicado.", fid)
            continue
        if engine is None:
            engine = daily._build_engine()  # noqa: SLF001
            table = daily._player_table()  # noqa: SLF001
        confirmed += int(_confirm_xi(rep, teams_lu, engine, table))
    return {"in_window": len(candidates), "attempts": attempts, "confirmed": confirmed}


def _confirm_xi(rep: dict, teams_lu: list[dict], engine: Any, table: pd.DataFrame) -> bool:
    """Recalcula props con el XI confirmado, guarda deltas y re-publica el reporte."""
    fid = int(rep["fixture_id"])
    home, away = str(rep["home"]), str(rep["away"])
    ko_match = bool(rep.get("is_knockout"))
    when = pd.Timestamp(str(rep["kickoff_utc"])[:10])
    pred = engine.predict_match(home, away, when=when)
    ko = proj.knockout_markets(pred["pmfs"]) if ko_match else None
    p_et = ko["p_prorroga"] if ko else 0.0
    te_factor = 1.0 + p_et * proj.ET_FRACTION * proj.ET_FATIGUE if ko else 1.0
    horizon = "120" if ko_match else "90"

    props_conf: dict[str, list[dict]] = {}
    deltas: dict[str, list[dict]] = {}
    for side, name in (("h", home), ("a", away)):
        xi = {int(p["player"]["id"]) for t in teams_lu
              if t["team"]["name"] == name for p in t.get("startXI", [])}
        before = daily._props_for(table, name, pred["means"], side,  # noqa: SLF001
                                  horizon=horizon, te_factor=te_factor)
        after = daily._props_for(table, name, pred["means"], side,  # noqa: SLF001
                                 horizon=horizon, te_factor=te_factor, lineup=xi or None)
        side_key = "home" if side == "h" else "away"
        props_conf[side_key] = _props_records(after)
        deltas[side_key] = _top_deltas(before, after)
        if after.empty:
            continue
        for _, r in after.head(TOP_PROPS_LOGGED).iterrows():
            store.ft_log_prediction(
                fixture_id=fid, match=f"{home} vs {away}", market="sot",
                player_id=int(r["player_id"]), player_name=str(r["player_name"]),
                pred_mean=float(r.get("mu_sot", 0)), notes="xi_confirmado",
            )

    row = dict(rep)
    payload = dict(row.get("payload") or {})
    payload["props"] = props_conf
    row["payload"] = payload
    row["xi_confirmed"] = True
    row["deltas"] = deltas
    row["updated_at"] = datetime.now(UTC).isoformat()
    store.save_daily_report(row)
    logger.info("lineups: XI confirmado para %s vs %s (fixture %d).", home, away, fid)
    return True


def _top_deltas(before: pd.DataFrame, after: pd.DataFrame, top: int = TOP_DELTAS) -> list[dict]:
    """Top |Δμ remates| por jugador entre el XI probable y el confirmado."""
    if before.empty or after.empty or "mu_shots" not in before.columns:
        return []
    merged = before.merge(after, on="player_id", suffixes=("_ant", "_conf"))
    if merged.empty:
        return []
    merged["delta"] = merged["mu_shots_conf"] - merged["mu_shots_ant"]
    order = merged["delta"].abs().sort_values(ascending=False).index
    out = []
    for _, r in merged.reindex(order).head(top).iterrows():
        out.append({
            "player_id": int(r["player_id"]),
            "player_name": str(r["player_name_ant"]),
            "mu_shots_antes": round(float(r["mu_shots_ant"]), 4),
            "mu_shots_conf": round(float(r["mu_shots_conf"]), 4),
            "delta": round(float(r["delta"]), 4),
        })
    return out


# ---------------------------------------------------------------------------
# settle — liquidación contra resultados reales (espejo cloud de cmd_post_day)
# ---------------------------------------------------------------------------

def run_settle(date_str: str | None = None) -> dict:
    """Liquida props + mercados de equipo del día y sube los datos nuevos."""
    return _run_job("settle", lambda: _settle(date_str))


def _settle(date_str: str | None) -> dict:
    key = get_settings().api_football_key
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    fixtures = [f for f in daily._day_fixtures(key, date_str, force=True)  # noqa: SLF001
                if f["fixture"]["status"]["short"] in nt_data.FINISHED_STATUSES]
    if not fixtures:
        return {"date": date_str, "fixtures": 0, "settled": 0}
    nt_data.fetch_fixture_details(key, [int(f["fixture"]["id"]) for f in fixtures])

    total = 0
    nt_rows: list[dict] = []
    pm_rows: list[dict] = []
    for fx in fixtures:
        fid = int(fx["fixture"]["id"])
        raw = players_wc.fetch_fixture_players(key, fid)
        real_rows = players_wc.parse_fixture_players(raw, fixture_id=fid)
        real = {r["player_id"]: r for r in real_rows}
        detail = _fixture_detail(fid)
        trow = nt_data.fixture_to_row(detail) if detail else None
        actuals = _team_actuals(fx, trow)

        settles: list[dict] = []
        for row in store.ft_pending(fid):
            outcome = _settlement_for(row, real, actuals)
            if outcome is not None:
                settles.append({"id": row["id"], "actual": outcome[0], "brier": outcome[1]})
        total += store.ft_settle_rows(settles)

        if trow is not None:
            nt_rows.append({"match_id": str(trow["match_id"]), "payload": _jsonable(trow)})
        pm_rows.extend({"id": f"{fid}_{r['player_id']}", "payload": _jsonable(r)}
                       for r in real_rows)

    store.upsert("nt_matches", nt_rows, on_conflict="match_id")
    store.upsert("player_matches", pm_rows, on_conflict="id")
    backed = store.daily_backup()
    return {"date": date_str, "fixtures": len(fixtures), "settled": total,
            "backup_rows": backed}


def _settlement_for(
    row: dict, real: dict[int, dict], actuals: dict
) -> tuple[float, float | None] | None:
    """(actual, brier) de una fila pendiente — MISMA matemática que forward_test/log."""
    if int(row.get("player_id") or 0) != 0:
        return ft_log._actual_and_brier(row, real)  # noqa: SLF001 — reuso deliberado
    out = ft_log._team_actual(str(row["market"]), row.get("line"), actuals)  # noqa: SLF001
    if out is None:
        return None
    actual, hit = out
    brier = None
    if row.get("pred_prob") is not None and hit is not None:
        brier = (float(row["pred_prob"]) - hit) ** 2
    return actual, brier


def _fixture_detail(fid: int) -> dict | None:
    """Detalle cacheado por fetch_fixture_details (None si falta o es stub)."""
    path = nt_data.NT_CACHE_DIR / f"fixture_{fid}.json"
    if not path.exists():
        return None
    detail = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(detail, dict) or detail.get("_missing"):
        return None
    return detail


def _team_actuals(fx: dict, trow: dict | None) -> dict:
    """Totales reales para liquidar team_* (misma armado que daily.cmd_post_day)."""
    if trow is None:
        return {}
    gh, ga = int(trow["home_score"]), int(trow["away_score"])
    winner_h = (fx.get("teams", {}).get("home", {}) or {}).get("winner")
    return {
        "goals_total": gh + ga,
        "corners_total": int(trow["corners_h"]) + int(trow["corners_a"]),
        "yellows_total": int(trow["yellows_h"]) + int(trow["yellows_a"]),
        "shots_total": int(trow["shots_h"]) + int(trow["shots_a"]),
        "sot_total": int(trow["sot_h"]) + int(trow["sot_a"]),
        "result": "home" if gh > ga else ("away" if ga > gh else "draw"),
        "btts": 1.0 if (gh > 0 and ga > 0) else 0.0,
        # ganador del fixture (incluye ET/pens si los hubo) = el que avanza
        "advanced": ("home" if winner_h is True
                     else ("away" if winner_h is False else None)),
    }


# ---------------------------------------------------------------------------
# weekly — resumen del forward-test (cálculo compartido con GET /wc/forward-test)
# ---------------------------------------------------------------------------

def run_weekly() -> dict:
    """Backup + resumen del forward-test; el detalle queda en job_runs."""
    return _run_job("weekly", _weekly)


def _weekly() -> dict:
    rows = store.select("props_log", {"order": "id.asc"})
    stats = compute_forward_test(rows)
    backed = store.daily_backup()
    s, ev = stats["summary"], stats["ev"]
    return {
        "total": s["total"], "settled": s["settled"], "pending": s["pending"],
        "brier_mean": s["brier_mean"], "mae": s["mae"],
        "n_con_cuota": ev["n_con_cuota"],
        "ev_teorico_medio": ev["ev_teorico_medio"],
        "roi_stake_plano": ev["roi_stake_plano"],
        "worst_markets": [m["market"] for m in stats["worst_markets"]],
        "backup_rows": backed,
    }


def _row_hit(row: dict) -> float | None:
    """Acierto 0/1 de una fila liquidada (None si el mercado no es binarizable)."""
    actual = row.get("actual")
    if actual is None:
        return None
    market = str(row.get("market") or "")
    line = row.get("line")
    if market.startswith("team_"):
        body = market[len("team_"):]
        for fam in ("goals", "corners", "yellows", "shots", "sot"):
            if body.startswith(f"{fam}_ou"):
                if line is None:
                    return None
                return 1.0 if float(actual) > float(line) else 0.0
        return float(actual)  # 1x2 / btts / se_clasifica: actual ES el acierto
    if market in ft_log.BINARY_MARKETS:
        return float(actual)
    if line is not None:  # conteo con línea (over)
        return 1.0 if float(actual) > float(line) else 0.0
    return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def compute_forward_test(rows: list[dict]) -> dict:
    """Resumen del forward-test desde filas de props_log (puro, sin red).

    ROI a stake plano por fila liquidada con cuota: acierto → odds−1, si no → −1.
    EV teórico por fila con cuota y pred_prob: pred_prob·odds − 1.
    Calibración por mercado: pred_avg vs real_avg (frecuencia real) y su gap.
    """
    settled = [r for r in rows if r.get("settled_at")]
    briers = [float(r["brier"]) for r in rows if r.get("brier") is not None]
    maes = [abs(float(r["pred_mean"]) - float(r["actual"])) for r in rows
            if r.get("pred_mean") is not None and r.get("actual") is not None]

    grouped: dict[str, dict] = {}
    for r in rows:
        m = str(r.get("market") or "?")
        g = grouped.setdefault(m, {"market": m, "n": 0, "n_settled": 0,
                                   "_briers": [], "_pred": [], "_real": []})
        g["n"] += 1
        if r.get("settled_at"):
            g["n_settled"] += 1
        if r.get("brier") is not None:
            g["_briers"].append(float(r["brier"]))
        hit = _row_hit(r)
        if r.get("settled_at") and r.get("pred_prob") is not None and hit is not None:
            g["_pred"].append(float(r["pred_prob"]))
            g["_real"].append(hit)

    by_market: list[dict] = []
    for g in grouped.values():
        pred_avg = _mean(g.pop("_pred"))
        real_avg = _mean(g.pop("_real"))
        g["brier"] = _mean(g.pop("_briers"))
        g["pred_avg"] = pred_avg
        g["real_avg"] = real_avg
        g["gap"] = (round(pred_avg - real_avg, 4)
                    if pred_avg is not None and real_avg is not None else None)
        by_market.append(g)
    by_market.sort(key=lambda g: -g["n"])
    worst = sorted(
        (g for g in by_market
         if g["gap"] is not None and g["n_settled"] >= MIN_SETTLED_FOR_WORST),
        key=lambda g: -abs(g["gap"]),
    )[:3]

    with_odds = [r for r in rows if r.get("odds") is not None]
    evs = [float(r["pred_prob"]) * float(r["odds"]) - 1.0
           for r in with_odds if r.get("pred_prob") is not None]
    rois: list[float] = []
    for r in with_odds:
        if not r.get("settled_at"):
            continue
        hit = _row_hit(r)
        if hit is None:
            continue
        rois.append(float(r["odds"]) - 1.0 if hit > 0.5 else -1.0)

    return {
        "summary": {
            "total": len(rows),
            "settled": len(settled),
            "pending": len(rows) - len(settled),
            "brier_mean": _mean(briers),
            "mae": _mean(maes),
        },
        "by_market": by_market,
        "worst_markets": worst,
        "ev": {
            "n_con_cuota": len(with_odds),
            "n_liquidadas_con_cuota": len(rois),
            "ev_teorico_medio": _mean(evs),
            "roi_stake_plano": _mean(rois),
        },
    }
