"""Jobs de automatización MLB (M4): espejo béisbol del pipeline cloud del Mundial.

Dos jobs sobre el MISMO storage compartido (wc/store.py → Supabase), con
sport='mlb' en daily_reports y props_log (los mercados MLB usan nombres
disjuntos — prefijo mlb_/ks — así que nunca colisionan con los del Mundial):

  run_mlb_daily   Por cada juego programado del día (statsapi, gratis, cacheado):
                  predicción unificada (wc/mlb_engine) + mercados (mlb_projection)
                  + props point-in-time (Ks de ambos abridores, hits/HR del
                  ÚLTIMO lineup conocido) → payload web + forward-test.
  run_mlb_settle  Liquida los juegos Final del día contra el linescore real
                  (runs / F5) y el boxscore (Ks/hits/HR por jugador) + backup.

Cuotas: get_mlb_odds lee odds-api.io (liga usa-mlb, Bet365) SOLO para mostrar —
acá no se apuesta nada. DEFENSIVO: cualquier shape inesperado → {} y log.

REUSO, no duplicación: la matemática vive en research/mlb.py (vía el engine),
markets/mlb_projection.py y players/mlb_props.py — acá solo se orquesta.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from mundial_bot.collectors import mlb_data, mlb_players
from mundial_bot.collectors.odds_oddspapi import ODDSAPI_BASE, _norm
from mundial_bot.config import get_settings
from mundial_bot.markets import mlb_projection as mlb_proj
from mundial_bot.players import mlb_props
from mundial_bot.research.distributions import p_over
from mundial_bot.wc import store
from mundial_bot.wc.jobs import _jsonable, _run_job

logger = logging.getLogger(__name__)

SPORT = "mlb"
PENDING_STATES = {"S", "P"}   # codedGameState de statsapi: Scheduled / Pre-Game
FINAL_STATE = "F"
TOP_BATTERS = 6               # bateadores por equipo al payload y forward-test
KS_LINE = 5.5                 # línea principal del prop de Ks
KS_ALT_LINE = 6.5
TOTAL_LINE = 8.5              # líneas de referencia del forward-test de equipo
F5_LINE = 4.5
RUN_LINE = 1.5
SCHEDULE_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# Schedule del día (statsapi, cache-primero en data/mlb_cache/day_{date}.json)
# ---------------------------------------------------------------------------

def _day_schedule(date_str: str, *, force: bool = False) -> list[dict]:
    """Juegos MLB del día con probables, venue y linescore (para el settle)."""
    mlb_data.MLB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = mlb_data.MLB_CACHE_DIR / f"day_{date_str}.json"
    if f.exists() and not force:
        d = json.loads(f.read_text(encoding="utf-8"))
    else:
        r = requests.get(f"{mlb_data.MLB_API}/schedule", params={
            "sportId": 1, "date": date_str, "hydrate": mlb_data.HYDRATE,
        }, timeout=SCHEDULE_TIMEOUT_S)
        r.raise_for_status()
        d = r.json()
        f.write_text(json.dumps(d), encoding="utf-8")
    return [g for day in d.get("dates", []) for g in day.get("games", [])]


def _build_engine():
    from mundial_bot.wc.mlb_engine import MlbLiveEngine
    return MlbLiveEngine()


# ---------------------------------------------------------------------------
# Props point-in-time (Ks de abridores + hits/HR del último lineup conocido)
# ---------------------------------------------------------------------------

def _starter_props(game: dict, as_of: pd.Timestamp) -> list[dict]:
    """Ks de ambos abridores probables: μ, var y P(over 5.5/6.5)."""
    out: list[dict] = []
    for side in ("home", "away"):
        pp = (game["teams"][side].get("probablePitcher") or {})
        pid = pp.get("id")
        if pid is None:
            continue
        try:
            mu, _var, pmf = mlb_props.pitcher_ks_distribution(int(pid), as_of=as_of)
        except requests.RequestException as exc:
            logger.warning("MLB Ks: sin gamelog de %s (%s): %s",
                           pp.get("fullName"), pid, exc)
            continue
        out.append({
            "player_id": int(pid),
            "player_name": pp.get("fullName") or "",
            "side": side,
            "mu_ks": round(float(mu), 4),
            "p_over_5.5": round(p_over(pmf, KS_LINE), 4),
            "p_over_6.5": round(p_over(pmf, KS_ALT_LINE), 4),
        })
    return out


def _last_lineup(hist: pd.DataFrame, team: str) -> list[dict]:
    """Último lineup TITULAR conocido del equipo (boxscore de su último juego FT)."""
    if hist.empty:
        return []
    rows = hist[(hist["home_team"] == team) | (hist["away_team"] == team)]
    if rows.empty:
        return []
    last = rows.sort_values("date").iloc[-1]
    side = "home" if last["home_team"] == team else "away"
    try:
        return mlb_players.fetch_game_lineup(int(last["game_pk"])).get(side, [])
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("MLB: sin lineup previo de %s (game %s): %s",
                       team, last["game_pk"], exc)
        return []


def _batter_gamelog(pid: int, as_of: pd.Timestamp) -> list[dict]:
    """Gamelog de bateo de las temporadas as_of y anterior (cache-primero)."""
    out: list[dict] = []
    for season in (as_of.year, as_of.year - 1):
        try:
            out.extend(mlb_players.fetch_batter_gamelog(pid, season))
        except requests.RequestException as exc:
            logger.warning("MLB: sin gamelog de bateo %s/%s: %s", pid, season, exc)
    return out


def _batter_props(team: str, team_hits_mean: float, hist: pd.DataFrame,
                  as_of: pd.Timestamp) -> list[dict]:
    """Top-N bateadores por μ de hits (reparto coherente del total del unificado)."""
    lineup = _last_lineup(hist, team)
    if not lineup:
        return []
    rates = {int(e["person_id"]): _batter_gamelog(int(e["person_id"]), as_of)
             for e in lineup}
    df = mlb_props.batter_hits_props(team_hits_mean, lineup, rates, as_of=as_of)
    if df.empty:
        return []
    out: list[dict] = []
    for _, r in df.sort_values("mu_hits", ascending=False).head(TOP_BATTERS).iterrows():
        pid = int(r["person_id"])
        p_hr = mlb_props.batter_hr_prob(
            pid, as_of=as_of, batting_order=int(r["batting_order"]),
            gamelog=rates.get(pid, []),
        )
        out.append({
            "player_id": pid,
            "player_name": str(r["name"]),
            "batting_order": int(r["batting_order"]),
            "mu_hits": round(float(r["mu_hits"]), 4),
            "p_hit_1plus": round(float(r["p_hit_1plus"]), 4),
            "p_hr": round(float(p_hr), 4),
        })
    return out


# ---------------------------------------------------------------------------
# mlb_daily — payload web + predicciones al forward-test
# ---------------------------------------------------------------------------

def run_mlb_daily(date_str: str | None = None) -> dict:
    """Payload web + predicciones de cada juego programado del día en Supabase."""
    return _run_job("mlb_daily", lambda: _daily(date_str))


def _daily(date_str: str | None) -> dict:
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    games = [g for g in _day_schedule(date_str)
             if ((g.get("status") or {}).get("codedGameState") or "") in PENDING_STATES]
    if not games:
        return {"date": date_str, "matches": 0, "predictions": 0}
    engine = _build_engine()
    as_of = pd.Timestamp(date_str)

    total_logged = 0
    for g in games:
        teams = g["teams"]
        home = teams["home"]["team"]["name"]
        away = teams["away"]["team"]["name"]
        venue = (g.get("venue") or {}).get("name") or ""
        pk = int(g["gamePk"])
        sp_h = teams["home"].get("probablePitcher") or {}
        sp_a = teams["away"].get("probablePitcher") or {}

        pred = engine.predict_game(home, away, venue, sp_h.get("id"), sp_a.get("id"),
                                   when=as_of)
        markets = mlb_proj.project_all_mlb(pred["pmfs"])
        ks_props = _starter_props(g, as_of)
        batters_h = _batter_props(home, float(pred["means"]["hits_h"]),
                                  engine.history, as_of)
        batters_a = _batter_props(away, float(pred["means"]["hits_a"]),
                                  engine.history, as_of)

        store.save_daily_report({
            "fixture_id": pk,
            "sport": SPORT,
            "report_date": date_str,
            "kickoff_utc": g.get("gameDate"),
            "home": home,
            "away": away,
            "round": "MLB",
            "is_knockout": False,
            "payload": {
                "venue": venue,
                "starters": {
                    "home": {"id": sp_h.get("id"), "name": sp_h.get("fullName") or ""},
                    "away": {"id": sp_a.get("id"), "name": sp_a.get("fullName") or ""},
                },
                "means": _jsonable(pred["means"], ndigits=6),
                "pmfs": {q: _jsonable(p, ndigits=6) for q, p in pred["pmfs"].items()},
                "markets": _jsonable(markets, ndigits=6),
                "props": {"pitchers": ks_props,
                          "batters": {"home": batters_h, "away": batters_a}},
            },
            "xi_confirmed": False,
        })
        total_logged += _log_game_predictions(pk, home, away, markets, ks_props,
                                              batters_h + batters_a)
    return {"date": date_str, "matches": len(games), "predictions": total_logged}


def _log_game_predictions(pk: int, home: str, away: str, markets: dict,
                          ks_props: list[dict], batter_props: list[dict]) -> int:
    """Predicciones inmutables al forward-test (≥8 por juego, sport='mlb')."""
    match = f"{away} @ {home}"  # convención MLB: visita @ local
    now = datetime.now(UTC).isoformat(timespec="seconds")
    n = 0

    def log(market: str, *, player_id: int = 0, player_name: str = "-",
            prob: float | None = None, mean: float | None = None,
            line: float | None = None) -> None:
        nonlocal n
        n += int(store.ft_log_prediction(
            sport=SPORT, fixture_id=pk, match=match, market=market,
            player_id=player_id, player_name=player_name,
            pred_mean=mean, pred_prob=prob, line=line, notes=f"as_of={now}",
        ))

    ml = markets["moneyline"]
    log("mlb_ml_home", prob=float(ml["home"]))
    log("mlb_ml_away", prob=float(ml["away"]))
    log(f"mlb_total_ou_{TOTAL_LINE}", line=TOTAL_LINE,
        prob=float(markets["totales"][str(TOTAL_LINE)]["over"]))
    log(f"mlb_f5_ou_{F5_LINE}", line=F5_LINE,
        prob=float(markets["f5"]["totales"][str(F5_LINE)]["over"]))
    log(f"mlb_rl_home_{RUN_LINE}", line=RUN_LINE,
        prob=float(markets["run_line"][f"home_-{RUN_LINE}"]))

    for p in ks_props:
        log("ks", player_id=p["player_id"], player_name=p["player_name"],
            mean=p["mu_ks"], prob=p["p_over_5.5"], line=KS_LINE)
    for b in batter_props:
        log("mlb_hits", player_id=b["player_id"], player_name=b["player_name"],
            mean=b["mu_hits"])
        log("mlb_hr", player_id=b["player_id"], player_name=b["player_name"],
            prob=b["p_hr"])
    return n


# ---------------------------------------------------------------------------
# mlb_settle — liquidación contra linescore (equipo) y boxscore (jugadores)
# ---------------------------------------------------------------------------

def run_mlb_settle(date_str: str | None = None) -> dict:
    """Liquida los juegos Final del schedule-date (default: AYER en UTC —
    los juegos del día D terminan en la madrugada UTC de D+1)."""
    return _run_job("mlb_settle", lambda: _settle(date_str))


def _settle(date_str: str | None) -> dict:
    date_str = date_str or (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    games = [g for g in _day_schedule(date_str, force=True)
             if ((g.get("status") or {}).get("codedGameState") or "") == FINAL_STATE]
    if not games:
        return {"date": date_str, "fixtures": 0, "settled": 0}

    total = 0
    for g in games:
        pk = int(g["gamePk"])
        pending = [r for r in store.ft_pending(pk) if r.get("sport", "wc") == SPORT]
        if not pending:
            continue
        actuals = _game_actuals(g)
        box = _boxscore_stats(pk)
        settles: list[dict] = []
        for row in pending:
            outcome = _settlement_for(row, actuals, box)
            if outcome is not None:
                settles.append({"id": row["id"], "actual": outcome[0],
                                "brier": outcome[1]})
        total += store.ft_settle_rows(settles)
    backed = store.daily_backup()
    return {"date": date_str, "fixtures": len(games), "settled": total,
            "backup_rows": backed}


def _game_actuals(g: dict) -> dict | None:
    """Totales reales del juego desde el linescore (None si vino incompleto)."""
    season = int(str(g.get("season") or g.get("gameDate", "")[:4] or 0) or 0)
    row = mlb_data._game_row(g, season)  # noqa: SLF001 — reuso deliberado del parser
    if row is None:
        return None
    return {
        "runs_h": int(row["runs_h"]), "runs_a": int(row["runs_a"]),
        "total": int(row["runs_h"]) + int(row["runs_a"]),
        "f5_total": int(row["runs_f5_h"]) + int(row["runs_f5_a"]),
        "margin": int(row["runs_h"]) - int(row["runs_a"]),
        # sin empate en MLB: el Final siempre tiene ganador
        "winner": "home" if row["runs_h"] > row["runs_a"] else "away",
    }


def _mlb_team_actual(
    market: str, line: float | None, actuals: dict
) -> tuple[float, float | None] | None:
    """(valor_real, acierto_0_1) de un mercado de equipo MLB; None si no aplica.

    ``actuals``: {"total", "f5_total", "margin", "winner" ('home'/'away')}.
    """
    if market in ("mlb_ml_home", "mlb_ml_away"):
        hit = 1.0 if actuals.get("winner") == market.rsplit("_", 1)[1] else 0.0
        return hit, hit
    if market.startswith("mlb_total_ou"):
        if line is None:
            return None
        total = actuals["total"]
        return float(total), (1.0 if float(total) > float(line) else 0.0)
    if market.startswith("mlb_f5_ou"):
        if line is None:
            return None
        total = actuals["f5_total"]
        return float(total), (1.0 if float(total) > float(line) else 0.0)
    if market.startswith("mlb_rl_home"):
        if line is None:
            return None
        margin = actuals["margin"]
        return float(margin), (1.0 if float(margin) > float(line) else 0.0)
    return None


def _boxscore_stats(pk: int) -> dict[int, dict]:
    """{person_id: stats reales} desde el boxscore FINAL (mismo cache que lineups).

    ``pitched``/``batted`` distinguen a quien figuró pero no jugó (p. ej. el
    probable rascado): esas filas quedan pendientes, no se liquidan en 0.
    """
    f = mlb_players.PLAYERS_CACHE_DIR / f"lineup_{pk}.json"
    d = mlb_players._get_cached(  # noqa: SLF001 — mismo endpoint/cache que lineups
        f, f"{mlb_data.MLB_API}/game/{pk}/boxscore", {}, force=False)
    out: dict[int, dict] = {}
    for side in ("home", "away"):
        players = ((d.get("teams") or {}).get(side) or {}).get("players") or {}
        for p in players.values():
            pid = (p.get("person") or {}).get("id")
            if pid is None:
                continue
            stats = p.get("stats") or {}
            pitching = stats.get("pitching") or {}
            batting = stats.get("batting") or {}
            out[int(pid)] = {
                "pitched": bool(pitching),
                "batted": bool(batting),
                "strikeouts": int(pitching.get("strikeOuts") or 0),
                "hits": int(batting.get("hits") or 0),
                "home_runs": int(batting.get("homeRuns") or 0),
            }
    return out


def _settlement_for(
    row: dict, actuals: dict | None, box: dict[int, dict]
) -> tuple[float, float | None] | None:
    """(actual, brier) de una fila pendiente MLB; None si aún no se puede liquidar."""
    market = str(row.get("market") or "")
    pid = int(row.get("player_id") or 0)
    if pid == 0:
        if actuals is None:
            return None
        out = _mlb_team_actual(market, row.get("line"), actuals)
        if out is None:
            return None
        actual, hit = out
        return actual, _brier(row, hit)

    st = box.get(pid)
    if st is None:
        return None  # no figura en el boxscore: queda pendiente
    if market == "ks":
        if not st["pitched"]:
            return None
        actual = float(st["strikeouts"])
        hit = _over_hit(actual, row.get("line"))
    elif market == "mlb_hits":
        if not st["batted"]:
            return None
        actual = float(st["hits"])
        hit = _over_hit(actual, row.get("line"))
    elif market == "mlb_hr":
        if not st["batted"]:
            return None
        actual = 1.0 if st["home_runs"] >= 1 else 0.0
        hit = actual
    else:
        return None
    return actual, _brier(row, hit)


def _over_hit(actual: float, line) -> float | None:
    return None if line is None else (1.0 if actual > float(line) else 0.0)


def _brier(row: dict, hit: float | None) -> float | None:
    if row.get("pred_prob") is None or hit is None:
        return None
    return (float(row["pred_prob"]) - hit) ** 2


# ---------------------------------------------------------------------------
# Cuotas EN VIVO (odds-api.io, liga usa-mlb, Bet365) — solo lectura, defensivo
# ---------------------------------------------------------------------------

MLB_ODDS_LEAGUE = "usa-mlb"
MLB_ODDS_SPORT = "baseball"
MLB_BOOKMAKERS = "Bet365"
_EVENTS_TTL_S = 3600          # /events cacheado 1 h (mapea equipos → eventId)
_ODDS_TTL_S = 600             # /odds cacheado 10 min por juego
ODDS_TIMEOUT_S = 25

_events_cache: dict[str, tuple[float, list[dict]]] = {}
_odds_cache: dict[int, tuple[float, dict]] = {}


def _fmt_line(hdp) -> str:
    try:
        return f"{float(hdp):g}"
    except (TypeError, ValueError):
        return str(hdp)


def _mlb_events(key: str) -> list[dict]:
    now = time.time()
    cached = _events_cache.get("mlb")
    if cached and now - cached[0] < _EVENTS_TTL_S:
        return cached[1]
    r = requests.get(f"{ODDSAPI_BASE}/events", params={
        "sport": MLB_ODDS_SPORT, "league": MLB_ODDS_LEAGUE, "apiKey": key,
    }, timeout=ODDS_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    events = data if isinstance(data, list) else (data.get("data") or [])
    _events_cache["mlb"] = (now, events)
    return events


def _find_mlb_event(events: list[dict], home: str, away: str, date_str: str) -> dict | None:
    """Cruza por equipos Y fecha (las series MLB repiten equipos en días seguidos)."""
    ok_dates = {date_str}
    try:
        d0 = datetime.strptime(date_str, "%Y-%m-%d")
        ok_dates.add((d0 + timedelta(days=1)).strftime("%Y-%m-%d"))  # night games UTC
    except ValueError:
        pass
    target = {_norm(home), _norm(away)}
    fallback = None
    for ev in events:
        pair = {_norm(str(ev.get("home") or "")), _norm(str(ev.get("away") or ""))}
        if pair != target:
            continue
        ev_date = str(ev.get("date") or ev.get("starts")
                      or ev.get("commence_time") or "")[:10]
        if ev_date in ok_dates:
            return ev
        if not ev_date and fallback is None:
            fallback = ev  # sin campo de fecha: mejor que nada
    return fallback


def _fetch_event_odds(key: str, event_id) -> tuple[dict, str | None]:
    """(raw, nota) — con filtro de casas; si el plan lo rechaza, sin filtro."""
    try:
        r = requests.get(f"{ODDSAPI_BASE}/odds", params={
            "eventId": event_id, "bookmakers": MLB_BOOKMAKERS, "apiKey": key,
        }, timeout=ODDS_TIMEOUT_S)
        r.raise_for_status()
        return (r.json() or {}), None
    except requests.RequestException as exc:
        logger.warning("MLB odds: filtro de bookmakers falló (%s); pruebo sin filtro.", exc)
    try:
        r = requests.get(f"{ODDSAPI_BASE}/odds", params={
            "eventId": event_id, "apiKey": key,
        }, timeout=ODDS_TIMEOUT_S)
        r.raise_for_status()
        return (r.json() or {}), "sin filtro de casas: filtrado client-side"
    except requests.RequestException as exc:
        return {}, f"sin cuotas ahora: {exc}"


def _put_best(d: dict, key: str, raw_odd, book: str) -> None:
    try:
        odd = float(raw_odd)
    except (TypeError, ValueError):
        return
    if odd <= 1.0:
        return
    cur = d.get(key)
    if cur is None or odd > cur["odd"]:
        d[key] = {"odd": odd, "book": book}


def _map_mlb_odds(raw: dict, allowed_books: set[str] | None) -> dict:
    """Respuesta cruda de /odds → {moneyline, totales, run_line}. Defensivo:
    cualquier shape inesperado devuelve {} y loguea, nunca crashea."""
    out: dict = {}
    try:
        books = raw.get("bookmakers") or {}
        if not isinstance(books, dict):
            return {}
        for book, markets in books.items():
            if allowed_books and str(book).strip().lower() not in allowed_books:
                continue
            for market in markets or []:
                name = str(market.get("name") or "").strip().lower()
                rows = market.get("odds") or []
                if name in ("ml", "moneyline"):
                    ml = out.setdefault("moneyline", {})
                    for row in rows:
                        _put_best(ml, "home", row.get("home"), book)
                        _put_best(ml, "away", row.get("away"), book)
                elif name == "totals":
                    tot = out.setdefault("totales", {})
                    for row in rows:
                        d = tot.setdefault(_fmt_line(row.get("hdp")), {})
                        _put_best(d, "over", row.get("over"), book)
                        _put_best(d, "under", row.get("under"), book)
                elif name in ("spread", "run line", "runline"):
                    rl = out.setdefault("run_line", {})
                    for row in rows:
                        d = rl.setdefault(_fmt_line(row.get("hdp")), {})
                        _put_best(d, "home", row.get("home"), book)
                        _put_best(d, "away", row.get("away"), book)
    except Exception:  # noqa: BLE001 — defensivo por contrato: {} antes que crash
        logger.exception("get_mlb_odds: shape inesperado de odds-api.io")
        return {}
    return out


def get_mlb_odds(game_pk: int, home: str, away: str, date: str) -> dict:
    """Cuotas Bet365 actuales del juego (odds-api.io), cacheadas 10 min.

    Solo lectura para que el humano compare y decida — acá no se apuesta nada.
    Mismo shape que /wc/live-odds: {markets, fetched_at, books, cache_age_s}.
    """
    now = time.time()
    cached = _odds_cache.get(game_pk)
    if cached and now - cached[0] < _ODDS_TTL_S:
        return {**cached[1], "cache_age_s": int(now - cached[0])}

    body: dict = {
        "game_pk": game_pk,
        "fetched_at": datetime.now(UTC).isoformat(),
        "books": [MLB_BOOKMAKERS],
        "markets": {},
    }
    key = get_settings().oddspapi_key
    if not key:
        return {**body, "note": "ODDSPAPI_KEY no configurada", "cache_age_s": 0}
    try:
        events = _mlb_events(key)
    except requests.RequestException as exc:
        logger.warning("get_mlb_odds: sin eventos de odds-api.io: %s", exc)
        return {**body, "note": f"sin eventos: {exc}", "cache_age_s": 0}
    ev = _find_mlb_event(events, home, away, date)
    if ev is None or ev.get("id") is None:
        return {**body, "note": "evento no encontrado en odds-api.io",
                "cache_age_s": 0}

    raw, note = _fetch_event_odds(key, ev["id"])
    allowed = ({b.strip().lower() for b in MLB_BOOKMAKERS.split(",")}
               if note and note.startswith("sin filtro") else None)
    body["markets"] = _map_mlb_odds(raw, allowed)
    if note:
        body["note"] = note
    _odds_cache[game_pk] = (now, body)
    return {**body, "cache_age_s": 0}
