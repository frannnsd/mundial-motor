"""Construye el cache de datos para la web de estadísticas.

Baja UNA vez desde API-Football (plan Pro de Franco) y guarda JSON en ``data/web/``
para que la API sirva rápido y sin depender de la red en cada request:

  - ``teams.json``      → los 48 equipos del Mundial (id, nombre, escudo, país).
  - ``fixtures.json``   → todos los partidos (fecha, estado, ronda, árbitro, sede, escudos).
  - ``players/<id>.json`` → plantel de cada equipo con stats ricas por jugador.

Uso: ``python scripts/build_web_cache.py`` (1 vez/día alcanza).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from mundial_bot.config import get_settings
from mundial_bot.value.team_aliases import normalize_team

BASE = "https://v3.football.api-sports.io"
LEAGUE = 1
SEASON = 2026
TIMEOUT = 30
WEB_DIR = Path(__file__).resolve().parents[1] / "data" / "web"
PLAYERS_DIR = WEB_DIR / "players"


def _get(key: str, path: str, params: dict) -> dict:
    r = requests.get(
        f"{BASE}/{path}", headers={"x-apisports-key": key}, params=params, timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def _num(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_teams(key: str) -> list[dict]:
    raw = _get(key, "teams", {"league": LEAGUE, "season": SEASON})
    out = []
    for it in raw.get("response", []):
        t = it.get("team") or {}
        if not t.get("id"):
            continue
        out.append({
            "id": int(t["id"]),
            "name": t.get("name"),
            "model_name": normalize_team(t.get("name") or ""),
            "code": t.get("code"),
            "country": t.get("country"),
            "logo": t.get("logo"),
        })
    return out


def aggregate_player(item: dict) -> dict | None:
    """Agrega las stats de un jugador a lo largo de la temporada (club + selección)."""
    p = item.get("player") or {}
    stats = item.get("statistics") or []
    agg = {
        "apps": 0, "minutes": 0, "goals": 0, "assists": 0, "shots": 0, "sot": 0,
        "passes": 0, "key_passes": 0, "dribbles": 0, "drib_succ": 0,
        "tackles": 0, "interceptions": 0, "blocks": 0, "duels": 0, "duels_won": 0,
        "fouls_drawn": 0, "fouls": 0, "yellow": 0, "red": 0,
        "pen_scored": 0, "pen_missed": 0,
    }
    rating_sum = 0.0
    rating_w = 0
    acc_sum = 0.0
    acc_w = 0
    position = None
    best_apps = -1
    team = None
    for st in stats:
        g = st.get("games") or {}
        apps = _num(g.get("appearences"))
        agg["apps"] += apps
        agg["minutes"] += _num(g.get("minutes"))
        if apps > best_apps:
            best_apps = apps
            position = g.get("position")
            team = (st.get("team") or {}).get("name")
        goals = st.get("goals") or {}
        agg["goals"] += _num(goals.get("total"))
        agg["assists"] += _num(goals.get("assists"))
        sh = st.get("shots") or {}
        agg["shots"] += _num(sh.get("total"))
        agg["sot"] += _num(sh.get("on"))
        pa = st.get("passes") or {}
        agg["passes"] += _num(pa.get("total"))
        agg["key_passes"] += _num(pa.get("key"))
        acc = _f(pa.get("accuracy"))
        if acc is not None and apps > 0:
            acc_sum += acc * apps
            acc_w += apps
        dr = st.get("dribbles") or {}
        agg["dribbles"] += _num(dr.get("attempts"))
        agg["drib_succ"] += _num(dr.get("success"))
        tk = st.get("tackles") or {}
        agg["tackles"] += _num(tk.get("total"))
        agg["interceptions"] += _num(tk.get("interceptions"))
        agg["blocks"] += _num(tk.get("blocks"))
        du = st.get("duels") or {}
        agg["duels"] += _num(du.get("total"))
        agg["duels_won"] += _num(du.get("won"))
        fo = st.get("fouls") or {}
        agg["fouls_drawn"] += _num(fo.get("drawn"))
        agg["fouls"] += _num(fo.get("committed"))
        ca = st.get("cards") or {}
        agg["yellow"] += _num(ca.get("yellow"))
        agg["red"] += _num(ca.get("red"))
        pe = st.get("penalty") or {}
        agg["pen_scored"] += _num(pe.get("scored"))
        agg["pen_missed"] += _num(pe.get("missed"))
        rt = _f(g.get("rating"))
        if rt is not None and apps > 0:
            rating_sum += rt * apps
            rating_w += apps
    if agg["apps"] <= 0:
        return None
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "photo": p.get("photo"),
        "age": p.get("age"),
        "nationality": p.get("nationality"),
        "position": position,
        "team": team,
        "rating": round(rating_sum / rating_w, 2) if rating_w else None,
        "pass_accuracy": round(acc_sum / acc_w, 1) if acc_w else None,
        **agg,
    }


def fetch_squad(key: str, team_id: int) -> list[dict]:
    players: list[dict] = []
    page = 1
    while page <= 6:
        raw = _get(key, "players", {"team": team_id, "season": SEASON, "page": page})
        for it in raw.get("response", []):
            pl = aggregate_player(it)
            if pl:
                players.append(pl)
        paging = raw.get("paging") or {}
        if page >= _num(paging.get("total")):
            break
        page += 1
        time.sleep(0.2)
    players.sort(key=lambda p: (-(p["goals"] or 0), -(p["apps"] or 0)))
    return players


def fetch_fixtures(key: str) -> list[dict]:
    raw = _get(key, "fixtures", {"league": LEAGUE, "season": SEASON})
    out = []
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
            "id": fx.get("id"),
            "date": fx.get("date"),
            "status": stt.get("short"),
            "round": lg.get("round"),
            "referee": fx.get("referee"),
            "venue": vn.get("name"),
            "city": vn.get("city"),
            "home": {
                "id": home.get("id"), "name": home.get("name"), "logo": home.get("logo"),
                "model_name": normalize_team(home.get("name") or ""),
            },
            "away": {
                "id": away.get("id"), "name": away.get("name"), "logo": away.get("logo"),
                "model_name": normalize_team(away.get("name") or ""),
            },
            "home_goals": go.get("home"),
            "away_goals": go.get("away"),
        })
    return out


def main() -> None:
    s = get_settings()
    key = s.api_football_key
    if not key:
        raise SystemExit("Falta API_FOOTBALL_KEY en .env")
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

    teams = fetch_teams(key)
    _write(WEB_DIR / "teams.json", teams)
    print(f"teams: {len(teams)}", flush=True)

    fixtures = fetch_fixtures(key)
    _write(WEB_DIR / "fixtures.json", fixtures)
    print(f"fixtures: {len(fixtures)}", flush=True)

    for i, t in enumerate(teams, 1):
        try:
            squad = fetch_squad(key, t["id"])
        except Exception as e:  # noqa: BLE001
            print(f"  squad {t['name']} ERROR {e}", flush=True)
            squad = []
        _write(PLAYERS_DIR / f"{t['id']}.json", squad)
        print(f"[{i}/{len(teams)}] {t['name']}: {len(squad)} players", flush=True)
        time.sleep(0.25)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
