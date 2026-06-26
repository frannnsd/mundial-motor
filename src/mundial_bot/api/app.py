"""API FastAPI que expone el motor del bot a la web de estadísticas.

Sirve los datos cacheados (equipos/jugadores/fixtures) + el cerebro matemático
(simulador, análisis de mercados) + el backtest + el chat con IA. La web Next.js
consume estos endpoints; nada de lógica se reimplementa en JavaScript.

El cerebro (Elo + Dixon-Coles + córners/tarjetas/tiros) tarda ~25s en entrenar:
se carga UNA vez al arrancar y queda en memoria.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mundial_bot.config import CACHE_DIR, DATA_DIR, get_settings

logger = logging.getLogger("mundial_api")
logging.basicConfig(level=logging.INFO)

WEB_DIR = CACHE_DIR.parent / "web"
PLAYERS_DIR = WEB_DIR / "players"
TEAM_STATS_CSV = CACHE_DIR / "team_match_stats.csv"

# Metadatos de las stats por jugador (para el filtro configurable del front).
STAT_KEYS = [
    {"key": "apps", "label": "Partidos", "group": "General"},
    {"key": "minutes", "label": "Minutos", "group": "General"},
    {"key": "rating", "label": "Rating", "group": "General"},
    {"key": "goals", "label": "Goles", "group": "Ataque"},
    {"key": "assists", "label": "Asistencias", "group": "Ataque"},
    {"key": "shots", "label": "Remates", "group": "Ataque"},
    {"key": "sot", "label": "Tiros al arco", "group": "Ataque"},
    {"key": "key_passes", "label": "Pases clave", "group": "Ataque"},
    {"key": "dribbles", "label": "Regates", "group": "Ataque"},
    {"key": "pen_scored", "label": "Penales", "group": "Ataque"},
    {"key": "passes", "label": "Pases", "group": "Pase"},
    {"key": "pass_accuracy", "label": "Precisión pase %", "group": "Pase"},
    {"key": "tackles", "label": "Barridas", "group": "Defensa"},
    {"key": "interceptions", "label": "Intercepciones", "group": "Defensa"},
    {"key": "blocks", "label": "Bloqueos", "group": "Defensa"},
    {"key": "duels", "label": "Duelos", "group": "Defensa"},
    {"key": "duels_won", "label": "Duelos ganados", "group": "Defensa"},
    {"key": "fouls", "label": "Faltas cometidas", "group": "Disciplina"},
    {"key": "fouls_drawn", "label": "Faltas recibidas", "group": "Disciplina"},
    {"key": "yellow", "label": "Amarillas", "group": "Disciplina"},
    {"key": "red", "label": "Rojas", "group": "Disciplina"},
]

TEAM_STAT_COLS = (
    "corners_for", "corners_against", "cards", "fouls", "shots", "sot_for", "sot_against",
)


class _State:
    brain = None
    teams: list[dict] = []
    teams_by_id: dict[int, dict] = {}
    teams_by_model: dict[str, dict] = {}
    fixtures: list[dict] = []
    team_aggs: dict[str, dict] = {}
    backtest_cache: dict | None = None
    injuries_cache: dict = {}
    frozen: bool = False


STATE = _State()


def _norm_txt(s: str) -> str:
    n = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _last_name(s: str) -> str:
    toks = _norm_txt(s).replace(".", " ").split()
    return toks[-1] if toks else ""


def _match_injuries(fixture_id: int | None) -> dict[str, list[dict]]:
    """Bajas (lesionados/suspendidos) por equipo (modelo) para un partido. Cacheado."""
    if not fixture_id:
        return {}
    if fixture_id in STATE.injuries_cache:
        return STATE.injuries_cache[fixture_id]
    settings = get_settings()
    out: dict[str, list[dict]] = {}
    if settings.has_api_football:
        try:
            from mundial_bot.collectors.injuries import fetch_injuries
            inj = fetch_injuries(settings.api_football_key, fixture_id=fixture_id)
            out = {
                team: [{"player": x.player, "reason": x.reason or x.kind or "Baja"} for x in lst]
                for team, lst in inj.items()
            }
        except Exception:  # noqa: BLE001
            out = {}
    STATE.injuries_cache[fixture_id] = out
    return out


def _extract_teams(message: str) -> list[str]:
    """Encuentra hasta 2 equipos del Mundial mencionados en un texto libre (es/en)."""
    from mundial_bot.brain import SPANISH_TEAMS

    msg = _norm_txt(message)
    candidates: list[tuple[int, str]] = []
    for t in STATE.teams:
        aliases = {t["name"], t["model_name"]}
        for sp, en in SPANISH_TEAMS.items():
            if en == t["model_name"]:
                aliases.add(sp)
        for al in aliases:
            n = _norm_txt(al)
            if len(n) < 3:
                continue
            m = re.search(r"\b" + re.escape(n) + r"\b", msg)
            if m:
                candidates.append((m.start(), t["model_name"]))
                break
    candidates.sort()
    out: list[str] = []
    for _, name in candidates:
        if name not in out:
            out.append(name)
    return out[:2]


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_team_aggs() -> dict[str, dict]:
    if not TEAM_STATS_CSV.exists():
        return {}
    df = pd.read_csv(TEAM_STATS_CSV)
    aggs: dict[str, dict] = {}
    for team, g in df.groupby("team"):
        row = {"matches": int(len(g))}
        for col in TEAM_STAT_COLS:
            row[col] = round(float(g[col].mean()), 2) if col in g else None
        aggs[str(team)] = row
    return aggs


API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
FINISHED_STATUS = {"FT", "AET", "PEN"}


def _fetch_fixtures_fresh(key: str) -> list[dict]:
    """Baja los fixtures del Mundial con su estado/resultado ACTUAL (1 llamada a la API)."""
    from mundial_bot.value.team_aliases import normalize_team

    raw = requests.get(
        f"{API_FOOTBALL_BASE}/fixtures",
        headers={"x-apisports-key": key},
        params={"league": 1, "season": 2026},
        timeout=30,
    ).json()
    out: list[dict] = []
    for it in raw.get("response", []):
        fx = it.get("fixture") or {}
        lg = it.get("league") or {}
        tm = it.get("teams") or {}
        go = it.get("goals") or {}
        home = tm.get("home") or {}
        away = tm.get("away") or {}
        vn = fx.get("venue") or {}
        stt = fx.get("status") or {}
        out.append({
            "id": fx.get("id"), "date": fx.get("date"),
            "status": stt.get("short"), "round": lg.get("round"),
            "referee": fx.get("referee"), "venue": vn.get("name"), "city": vn.get("city"),
            "home": {"id": home.get("id"), "name": home.get("name"), "logo": home.get("logo"),
                     "model_name": normalize_team(home.get("name") or "")},
            "away": {"id": away.get("id"), "name": away.get("name"), "logo": away.get("logo"),
                     "model_name": normalize_team(away.get("name") or "")},
            "home_goals": go.get("home"), "away_goals": go.get("away"),
        })
    return out


def _refresh_live_data() -> dict:
    """Refresca fixtures + resultados del Mundial y limpia caches derivadas (backtest/bajas)."""
    settings = get_settings()
    out = {"fixtures": 0, "results": 0}
    if not settings.has_api_football:
        return out
    try:
        fx = _fetch_fixtures_fresh(settings.api_football_key)
        if fx:
            (WEB_DIR / "fixtures.json").write_text(
                json.dumps(fx, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            STATE.fixtures = fx
            out["fixtures"] = len(fx)
    except Exception as e:  # noqa: BLE001
        logger.warning("Refresh de fixtures falló: %s", e)
    try:
        from mundial_bot.collectors.wc_results import build_cache as refresh_wc
        df = refresh_wc(settings.api_football_key)
        out["results"] = 0 if df is None else int(len(df))
    except Exception as e:  # noqa: BLE001
        logger.warning("Refresh de resultados falló: %s", e)
    STATE.backtest_cache = None
    STATE.injuries_cache = {}
    return out


async def _refresh_loop() -> None:
    """Cada 20 min refresca fixtures/resultados; cada ~2h recarga el cerebro con lo nuevo."""
    n = 0
    while True:
        await asyncio.sleep(1200)
        n += 1
        try:
            res = await asyncio.to_thread(_refresh_live_data)
            logger.info("Auto-refresh: %s", res)
            if n % 6 == 0 and not STATE.frozen:  # ~cada 2h: recargar con los resultados nuevos
                from mundial_bot.brain import load_brain
                STATE.brain = await asyncio.to_thread(load_brain)
                STATE.backtest_cache = None
                logger.info("Cerebro recargado con los resultados nuevos.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Auto-refresh falló: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando datos del cache...")
    STATE.teams = _load_json(WEB_DIR / "teams.json", [])
    STATE.teams_by_id = {t["id"]: t for t in STATE.teams}
    STATE.teams_by_model = {t["model_name"]: t for t in STATE.teams}
    STATE.fixtures = _load_json(WEB_DIR / "fixtures.json", [])
    STATE.team_aggs = _compute_team_aggs()
    from mundial_bot.brain import load_brain

    brain_pkl = DATA_DIR / "brain.pkl"
    if brain_pkl.exists():
        import pickle

        logger.info("Cargando cerebro congelado (arranque rápido)...")
        with brain_pkl.open("rb") as f:
            STATE.brain = pickle.load(f)
        STATE.frozen = True
    else:
        logger.info("Entrenando el cerebro (Elo + Dixon-Coles)... ~25s")
        STATE.brain = load_brain()
    logger.info("Cerebro listo. %d equipos, %d fixtures.", len(STATE.teams), len(STATE.fixtures))
    try:
        logger.info("Refresh inicial: %s", _refresh_live_data())
    except Exception as e:  # noqa: BLE001
        logger.warning("Refresh inicial falló: %s", e)
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()


app = FastAPI(title="Mundial Stats API", version="1.0.0", lifespan=lifespan)
# Permite localhost y cualquier IP de red local (para abrir desde el celu en la WiFi).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"http://(localhost|127\.0\.0\.1|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?"
        r"|https://[a-z0-9-]+\.vercel\.app"  # la web en producción (Vercel)
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "brain_loaded": STATE.brain is not None,
        "teams": len(STATE.teams),
        "fixtures": len(STATE.fixtures),
    }


@app.get("/stat-keys")
def stat_keys() -> list[dict]:
    return STAT_KEYS


def _team_card(t: dict) -> dict:
    """Equipo + sus promedios de equipo (forma agregada)."""
    agg = STATE.team_aggs.get(t["model_name"], {})
    return {**t, "agg": agg}


@app.get("/teams")
def teams() -> list[dict]:
    cards = [_team_card(t) for t in STATE.teams]
    cards.sort(key=lambda c: c["name"])
    return cards


def _team_fixtures(team_id: int) -> list[dict]:
    out = [f for f in STATE.fixtures if (f["home"]["id"] == team_id or f["away"]["id"] == team_id)]
    out.sort(key=lambda f: f.get("date") or "")
    return out


@app.get("/teams/{team_id}")
def team_detail(team_id: int) -> dict:
    t = STATE.teams_by_id.get(team_id)
    if not t:
        raise HTTPException(status_code=404, detail="Equipo no encontrado")
    players = _load_json(PLAYERS_DIR / f"{team_id}.json", [])
    return {
        **t,
        "agg": STATE.team_aggs.get(t["model_name"], {}),
        "players": players,
        "fixtures": _team_fixtures(team_id),
    }


@app.get("/teams/{team_id}/players")
def team_players(team_id: int) -> list[dict]:
    if team_id not in STATE.teams_by_id:
        raise HTTPException(status_code=404, detail="Equipo no encontrado")
    return _load_json(PLAYERS_DIR / f"{team_id}.json", [])


@app.get("/fixtures")
def fixtures() -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=3)  # ya pasó = no es "próximo"

    def is_upcoming(f: dict) -> bool:
        if f.get("status") in FINISHED_STATUS:
            return False
        d = f.get("date")
        if not d:
            return True
        try:
            return datetime.fromisoformat(str(d).replace("Z", "+00:00")) > cutoff
        except Exception:  # noqa: BLE001
            return True

    upcoming = [f for f in STATE.fixtures if is_upcoming(f)]
    played = [f for f in STATE.fixtures if f.get("status") in FINISHED_STATUS]
    upcoming.sort(key=lambda f: f.get("date") or "")
    played.sort(key=lambda f: f.get("date") or "", reverse=True)
    return {"upcoming": upcoming, "played": played}


def _resolve_team_arg(arg: str) -> dict | None:
    """Acepta team_id, model_name o nombre libre y devuelve el equipo del cache."""
    if arg.isdigit() and int(arg) in STATE.teams_by_id:
        return STATE.teams_by_id[int(arg)]
    if arg in STATE.teams_by_model:
        return STATE.teams_by_model[arg]
    resolved = STATE.brain.resolve(arg) if STATE.brain else None
    return STATE.teams_by_model.get(resolved) if resolved else None


@app.get("/match")
def match(home: str, away: str) -> dict:
    h = _resolve_team_arg(home)
    a = _resolve_team_arg(away)
    if not h or not a:
        raise HTTPException(status_code=404, detail="No pude resolver uno de los equipos")
    return {
        "home": _team_card(h),
        "away": _team_card(a),
    }


class SimRequest(BaseModel):
    home: str
    away: str
    fixture_id: int | None = None
    n: int = 8000
    neutral: bool = True
    auto_context: bool = True
    # overrides manuales (opcionales; se suman a lo detectado)
    knockout: bool | None = None
    referee: str | None = None
    rivalry: bool = False
    heat: bool = False
    altitude: bool = False
    home_motivation: str = "auto"
    away_motivation: str = "auto"


def _points_table() -> dict[str, dict]:
    """Tabla simple (puntos/PJ) de los partidos de grupo ya jugados, para la motivación."""
    table: dict[str, dict] = {}
    for f in STATE.fixtures:
        if f.get("status") not in ("FT", "AET", "PEN"):
            continue
        if "group" not in (f.get("round") or "").lower():
            continue
        hg, ag = f.get("home_goals"), f.get("away_goals")
        if hg is None or ag is None:
            continue
        h, a = f["home"]["model_name"], f["away"]["model_name"]
        for t in (h, a):
            table.setdefault(t, {"pts": 0, "gp": 0})
        table[h]["gp"] += 1
        table[a]["gp"] += 1
        if hg > ag:
            table[h]["pts"] += 3
        elif hg < ag:
            table[a]["pts"] += 3
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1
    return table


def _motivation(team: str, table: dict, round_str: str = "") -> str:
    """Heurística de motivación desde los puntos y la fecha.

    Caso clave: en la ÚLTIMA fecha de grupos, quien no esté ya clasificado (≤3 pts)
    se juega la clasificación → 'must_win' (se juega la vida). Ya con 6+ pts está
    prácticamente adentro → 'qualified' (administra).
    """
    row = table.get(team)
    if not row or row["gp"] < 1:
        return "normal"
    pts, gp = row["pts"], row["gp"]
    final_group = "3" in (round_str or "")  # "Group Stage - 3"
    if pts >= 6:
        return "qualified"
    if final_group or gp >= 2:
        return "must_win" if pts <= 3 else "normal"  # 0-3 pts en la última = a todo o nada
    if pts <= 0:
        return "must_win"
    return "normal"


def _fixture_by_id(fixture_id: int | None) -> dict | None:
    if fixture_id is None:
        return None
    return next((f for f in STATE.fixtures if f["id"] == fixture_id), None)


@app.post("/simulate")
def simulate_match(req: SimRequest) -> dict:
    if STATE.brain is None:
        raise HTTPException(status_code=503, detail="El cerebro todavía no cargó")
    from mundial_bot.sim.context import ALTITUDE_CITIES, HEAT_CITIES, RIVALRIES, build_context
    from mundial_bot.sim.match_sim import simulate

    fx = _fixture_by_id(req.fixture_id)
    referee = req.referee
    hm = req.home_motivation if req.home_motivation != "auto" else "normal"
    am = req.away_motivation if req.away_motivation != "auto" else "normal"
    knockout = bool(req.knockout)
    heat = req.heat
    altitude = req.altitude
    rivalry = req.rivalry

    if req.auto_context and fx:
        rnd = (fx.get("round") or "").lower()
        knockout = bool(rnd) and "group" not in rnd and "qualif" not in rnd
        city = (fx.get("city") or "").lower()
        heat = heat or city in HEAT_CITIES
        altitude = altitude or city in ALTITUDE_CITIES
        rivalry = rivalry or frozenset({req.home, req.away}) in RIVALRIES
        referee = referee or fx.get("referee")
        table = _points_table()
        if req.home_motivation == "auto":
            hm = _motivation(req.home, table, rnd)
        if req.away_motivation == "auto":
            am = _motivation(req.away, table, rnd)

    ctx = build_context(
        knockout=knockout, rivalry=rivalry, heat=heat, altitude=altitude,
        home_motivation=hm, away_motivation=am,
    )
    try:
        res = simulate(
            STATE.brain, req.home, req.away,
            n=max(1000, min(req.n, 50000)), neutral=req.neutral,
            referee=referee, knockout=knockout, context=ctx,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    res["context_detected"] = {
        "knockout": knockout, "heat": heat, "altitude": altitude, "rivalry": rivalry,
        "home_motivation": hm, "away_motivation": am, "referee": referee,
        "city": fx.get("city") if fx else None,
    }
    return res


@app.get("/markets")
def markets(home: str, away: str, fixture_id: int | None = None) -> dict:
    """Tablero COMPLETO de mercados (modelo % + cuota REAL de Bet365/Betano)."""
    if STATE.brain is None:
        raise HTTPException(status_code=503, detail="El cerebro todavía no cargó")
    from mundial_bot.models.market_book import build_market_book, real_odd

    b = STATE.brain
    fx = _fixture_by_id(fixture_id)
    rnd = (fx.get("round") or "").lower() if fx else ""
    knockout = bool(rnd) and "group" not in rnd and "qualif" not in rnd
    referee = fx.get("referee") if fx else None
    rh, ra = b.resolve(home), b.resolve(away)

    book = build_market_book(
        rh, ra, elo=b.models.elo, goals=b.models.goals,
        corners=b.corners, cards=b.cards, shots=b.shots, total_shots=b.total_shots,
        referee=referee, knockout=knockout,
    )

    odds: dict = {}
    settings = get_settings()
    if fx and settings.has_api_football:
        from mundial_bot.collectors.odds_af import fetch_odds, merge_odds
        try:
            odds = fetch_odds(settings.api_football_key, fixture_id, books=settings.preferred_books_set)
        except Exception:  # noqa: BLE001
            odds = {}
        if settings.has_oddspapi:
            try:
                from mundial_bot.collectors.odds_oddspapi import fetch_match_odds
                extra = fetch_match_odds(settings.oddspapi_key, rh, ra)
                if extra:
                    odds = merge_odds(odds, extra) if odds else extra
            except Exception:  # noqa: BLE001
                pass

    groups = []
    for market, sels in book.by_market().items():
        items = []
        for s in sels:
            ro = real_odd(s, odds)
            items.append({
                "pick": s.pick, "prob": round(s.prob, 4), "fair": s.fair,
                "push": round(s.push, 4), "note": s.note,
                "odd": round(ro[0], 2) if ro else None, "book": ro[1] if ro else None,
            })
        groups.append({"market": market, "selections": items})

    return {
        "header": {
            "home": book.home, "away": book.away,
            "home_xg": round(book.home_xg, 2), "away_xg": round(book.away_xg, 2),
            "p_home": round(book.p_home, 4), "p_draw": round(book.p_draw, 4), "p_away": round(book.p_away, 4),
            "elo": [round(book.elo_home, 4), round(book.elo_draw, 4), round(book.elo_away, 4)],
            "dc": [round(book.dc_home, 4), round(book.dc_draw, 4), round(book.dc_away, 4)],
        },
        "markets": groups,
        "has_odds": bool(odds),
    }


@app.get("/backtest")
def backtest() -> dict:
    if STATE.backtest_cache is None:
        bt_file = DATA_DIR / "backtest.json"
        if STATE.frozen and bt_file.exists():
            STATE.backtest_cache = json.loads(bt_file.read_text(encoding="utf-8"))
        else:
            from mundial_bot.backtest.sim_backtest import run_backtest

            logger.info("Corriendo backtest (entrena sin el Mundial)...")
            STATE.backtest_cache = run_backtest()
    return STATE.backtest_cache


def _player_projections(
    team_id: int, rival_model: str, team_xg: float, injured_last: set[str] | None = None
) -> list[dict]:
    """Proyección POR PARTIDO de cada jugador, ajustada por el rival.

    Remates/tiros al arco se escalan por cuánto concede el rival (factor defensivo).
    Los goles reparten el xG del equipo según la tasa de gol de cada jugador.
    El resto (barridas, faltas ganadas/cometidas, asistencias) es su tasa por partido.
    Asume titularidad (las alineaciones salen ~1h antes).
    """
    from mundial_bot.collectors.player_stats import opponent_factor

    b = STATE.brain
    players = _load_json(PLAYERS_DIR / f"{team_id}.json", [])
    sot_f = opponent_factor(b.shots, rival_model) if b.shots else 1.0
    sh_f = opponent_factor(b.total_shots, rival_model) if getattr(b, "total_shots", None) else 1.0
    injured_last = injured_last or set()
    elig = [
        p for p in players
        if (p.get("apps") or 0) >= 2 and _last_name(p.get("name", "")) not in injured_last
    ]
    gpg_total = sum((p["goals"] / p["apps"]) for p in elig if p["apps"]) or 1.0

    out: list[dict] = []
    for p in elig:
        ap = p["apps"] or 1
        exp_shots = p["shots"] / ap * sh_f
        exp_sot = p["sot"] / ap * sot_f
        exp_goals = team_xg * ((p["goals"] / ap) / gpg_total) if gpg_total > 0 else 0.0
        out.append({
            "id": p["id"], "name": p["name"], "photo": p.get("photo"), "position": p.get("position"),
            "exp_shots": round(exp_shots, 2), "p_shot": round(1 - math.exp(-exp_shots), 3),
            "exp_sot": round(exp_sot, 2), "p_sot": round(1 - math.exp(-exp_sot), 3),
            "exp_goals": round(exp_goals, 2), "p_goal": round(1 - math.exp(-exp_goals), 3),
            "exp_assists": round(p["assists"] / ap, 2),
            "exp_tackles": round(p["tackles"] / ap, 2),
            "exp_fouls_drawn": round(p["fouls_drawn"] / ap, 2),
            "exp_fouls": round(p["fouls"] / ap, 2),
        })
    out.sort(key=lambda r: -(r["exp_goals"] + r["exp_shots"] / 12))
    return out


@app.get("/match-players")
def match_players(home: str, away: str, fixture_id: int | None = None) -> dict:
    """Proyección por jugador para un partido (los dos equipos), ajustada por el rival."""
    if STATE.brain is None:
        raise HTTPException(status_code=503, detail="El cerebro todavía no cargó")
    b = STATE.brain
    th, ta = _resolve_team_arg(home), _resolve_team_arg(away)
    if not th or not ta:
        raise HTTPException(status_code=404, detail="No pude resolver los equipos")
    rh, ra = th["model_name"], ta["model_name"]
    home_xg = away_xg = 1.2
    if b.models.goals and b.models.goals.can_predict(rh, ra):
        try:
            _, home_xg, away_xg = b.models.goals.score_matrix(rh, ra, neutral=True)
        except Exception:  # noqa: BLE001
            pass
    inj = _match_injuries(fixture_id)
    home_inj, away_inj = inj.get(rh, []), inj.get(ra, [])
    home_last = {_last_name(x["player"]) for x in home_inj}
    away_last = {_last_name(x["player"]) for x in away_inj}
    return {
        "home": {"team": th["name"], "team_id": th["id"], "injured": home_inj,
                 "players": _player_projections(th["id"], ra, float(home_xg), home_last)},
        "away": {"team": ta["name"], "team_id": ta["id"], "injured": away_inj,
                 "players": _player_projections(ta["id"], rh, float(away_xg), away_last)},
    }


class ComboRequest(BaseModel):
    home: str
    away: str
    target: float
    fixture_id: int | None = None


@app.post("/build-combo")
def build_combo_ep(req: ComboRequest) -> dict:
    """Arma la combinada del partido que más se acerca a la cuota objetivo pedida."""
    if STATE.brain is None:
        raise HTTPException(status_code=503, detail="El cerebro todavía no cargó")
    b = STATE.brain
    rh, ra = b.resolve(req.home), b.resolve(req.away)
    if not b.models.goals or not b.models.goals.can_predict(rh, ra):
        raise HTTPException(status_code=400, detail="No tengo modelo de goles para este partido")

    odds: dict = {}
    fx = _fixture_by_id(req.fixture_id)
    settings = get_settings()
    if fx and settings.has_api_football:
        from mundial_bot.collectors.odds_af import fetch_odds
        try:
            odds = fetch_odds(settings.api_football_key, req.fixture_id, books=settings.preferred_books_set)
        except Exception:  # noqa: BLE001
            odds = {}

    from mundial_bot.sim.combo import build_combo
    target = max(1.2, min(float(req.target), 5000.0))
    return build_combo(b, rh, ra, target, odds=odds)


class LiveRequest(BaseModel):
    home: str
    away: str
    home_goals: int = 0
    away_goals: int = 0
    minute: float = 0


_LIVE_MARKETS = {"Ganador (1X2)", "Goles Más/Menos", "Ambos marcan", "Doble oportunidad"}


@app.post("/live")
def live_ep(req: LiveRequest) -> dict:
    """Panorama EN VIVO (mercados del resto del partido) + alertas de apuestas vivas."""
    if STATE.brain is None:
        raise HTTPException(status_code=503, detail="El cerebro todavía no cargó")
    b = STATE.brain
    goals = b.models.goals
    rh, ra = b.resolve(req.home), b.resolve(req.away)
    if not goals or not goals.can_predict(rh, ra):
        raise HTTPException(status_code=400, detail="No tengo modelo de goles para este partido")

    from mundial_bot.models.live import build_live_book

    book = build_live_book(
        rh, ra, goals=goals,
        home_goals=req.home_goals, away_goals=req.away_goals, minute=req.minute,
    )
    groups = []
    alerts = []
    for market, sels in book.by_market().items():
        if market not in _LIVE_MARKETS:
            continue
        groups.append({
            "market": market,
            "selections": [{"pick": s.pick, "prob": round(s.prob, 4), "fair": s.fair} for s in sels],
        })
        for s in sels:
            if 0.62 <= s.prob <= 0.93 and s.pick.strip().lower() != "empate":
                alerts.append({"pick": s.pick, "market": market, "prob": round(s.prob, 4), "fair": s.fair})
    alerts.sort(key=lambda a: -a["prob"])
    rem = int(max(0, 90 - min(req.minute, 90)))
    return {
        "home": rh, "away": ra,
        "home_goals": req.home_goals, "away_goals": req.away_goals, "minute": req.minute,
        "remaining_min": rem, "remaining_xg": round(book.home_xg + book.away_xg, 2),
        "result": {"home": round(book.p_home, 4), "draw": round(book.p_draw, 4), "away": round(book.p_away, 4)},
        "markets": groups,
        "alerts": alerts[:6],
    }


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    settings = get_settings()
    if not settings.has_anthropic:
        raise HTTPException(status_code=503, detail="Falta ANTHROPIC_API_KEY")
    import anthropic

    from mundial_bot.sim.match_sim import simulate

    brain = STATE.brain
    blocks: list[str] = []
    found = _extract_teams(req.message) if brain else []
    if len(found) >= 2:
        h, a = found[0], found[1]
        try:
            blocks.append("ANÁLISIS DEL MODELO (probabilidades por mercado):\n" + brain.full_analysis(h, a))
        except Exception:  # noqa: BLE001
            pass
        try:
            sim = simulate(brain, h, a, n=6000)
            blocks.append(
                f"SIMULACIÓN ({sim['n']} partidos): gana {sim['home']} {sim['result']['home']:.0%}, "
                f"empate {sim['result']['draw']:.0%}, gana {sim['away']} {sim['result']['away']:.0%}. "
                f"Marcador top {sim['top_scores'][0]['score']}. BTTS {sim['btts']:.0%}."
            )
        except Exception:  # noqa: BLE001
            pass

    system = (
        "Sos 'Apu', analista experto del Mundial 2026 que ayuda a Franco a decidir sus "
        "apuestas en Bet365. Hablás español argentino, directo y canchero. Franco decide; "
        "vos le das tu lectura clara del partido usando los NÚMEROS del modelo que te paso. "
        "No le manejes la plata ni le des sermones. No inventes datos: si no te pasé algo, "
        "decílo. Respondé conciso (máx ~150 palabras)."
    )
    user = req.message
    if blocks:
        user += "\n\n[Datos del modelo para tu respuesta]\n" + "\n\n".join(blocks)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=900, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {"reply": text}
