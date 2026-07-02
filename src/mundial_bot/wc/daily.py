"""Pipeline diario de operación en vivo del Mundial 2026.

Uso:  python -m mundial_bot.wc.daily <subcomando>

  pre-day [--date YYYY-MM-DD]     Reporte + registro de predicciones del día.
  pre-kickoff --fixture <id>      Recalcula props con el XI CONFIRMADO (delta).
  post-day [--date YYYY-MM-DD]    Liquida contra los resultados reales.
  weekly                          Reporte acumulado del forward-test.
  add-odds --fixture <id> --player <id|0> --market <m> --line <x> --odds <y>
                                  Carga a mano la línea/cuota de bet365.

Reglas: cache-primero (pre-day ≈ 1-2 llamadas: la lista del día; el motor y los
props usan el cache de disco); pre-kickoff 1 llamada de lineups SOLO dentro de la
ventana -60min→kickoff; toda predicción emitida queda logueada en el forward-test
con as_of en `notes`. El XI confirmado jamás se usa para predicciones anteriores
a su publicación (no-leakage, heredado de players/props).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from mundial_bot.config import get_settings
from mundial_bot.forward_test import log as ft
from mundial_bot.markets import projection as proj
from mundial_bot.wc.engine import WcEngine

API_BASE = "https://v3.football.api-sports.io"
NT_CACHE = Path(__file__).resolve().parents[3] / "data" / "nt_cache"
REPORTS_DAILY = Path(__file__).resolve().parents[3] / "reports" / "daily"
LINEUP_WINDOW_MIN = 60          # ventana para pedir lineups: -60min → kickoff
PROPS_STATS = ("shots", "sot", "goals", "yellow")  # totales que el motor modela
_KO_TOKENS = ("Round of", "Quarter", "Semi", "Final", "3rd Place", "Third Place")


def _get_cached(key: str, path: str, params: dict, cache_name: str, *, force: bool = False) -> dict:
    """GET con cache-primero en data/nt_cache (idéntica filosofía al resto del repo)."""
    NT_CACHE.mkdir(parents=True, exist_ok=True)
    f = NT_CACHE / f"{cache_name}.json"
    if f.exists() and not force:
        return json.loads(f.read_text(encoding="utf-8"))
    r = requests.get(f"{API_BASE}{path}", headers={"x-apisports-key": key},
                     params=params, timeout=30)
    r.raise_for_status()
    d = r.json()
    time.sleep(0.4)
    if d.get("response"):
        f.write_text(json.dumps(d), encoding="utf-8")
    return d


def _day_fixtures(key: str, date_str: str, *, force: bool = False) -> list[dict]:
    d = _get_cached(key, "/fixtures", {"league": 1, "season": 2026, "date": date_str},
                    f"day_{date_str}", force=force)
    return d.get("response", [])


def _is_knockout(fx: dict) -> bool:
    rnd = (fx.get("league", {}) or {}).get("round", "") or ""
    return any(tok in rnd for tok in _KO_TOKENS)


def _build_engine() -> WcEngine:
    from mundial_bot.collectors.nt_data import build_nt_match_table
    df = build_nt_match_table()
    if df.empty:
        raise SystemExit("Tabla de selecciones vacía: corré primero el collector nt_data.")
    return WcEngine(df)


def _player_table() -> pd.DataFrame:
    from mundial_bot.collectors.players_wc import build_player_match_table
    return build_player_match_table(get_settings().api_football_key)


def _team_totals(means: dict, side: str, factor: float = 1.0) -> dict:
    """Totales del motor → stats de props (en el MISMO horizonte, coherencia)."""
    m = {
        "shots": means[f"shots_{side}"], "sot": means[f"sot_{side}"],
        "goals": means[f"goals_{side}"], "yellow": means[f"yellows_{side}"],
    }
    return {k: (v * factor, v * factor * 1.3) for k, v in m.items()}


def _props_for(
    table: pd.DataFrame, team: str, means: dict, side: str,
    *, horizon: str = "90", te_factor: float = 1.0, lineup: set[int] | None = None,
) -> pd.DataFrame:
    from mundial_bot.players.shares import player_shares
    shares = player_shares(table, team)
    if shares.empty:
        return shares
    from mundial_bot.players.props import match_props
    return match_props(_team_totals(means, side, te_factor), shares,
                       horizon=horizon, lineup_confirmed=lineup)


def _fmt_props(df: pd.DataFrame, top: int = 8) -> list[str]:
    lines = ["| Jugador | min | μ remates | μ al arco | P(anota) | P(tarjeta) |",
             "|---|---|---|---|---|---|"]
    for _, r in df.head(top).iterrows():
        lines.append(
            f"| {r['player_name']} | {r['exp_minutes']:.0f} | {r.get('mu_shots', 0):.2f} "
            f"| {r.get('mu_sot', 0):.2f} | {r.get('p_scores', 0):.0%} "
            f"| {r.get('p_card', 0):.0%} |"
        )
    return lines


def _log_match_predictions(fx: dict, pred: dict, markets: dict, ko: dict | None,
                           props_h: pd.DataFrame, props_a: pd.DataFrame) -> int:
    """Registra ~15-25 predicciones clave del partido en el forward-test."""
    fid = int(fx["fixture"]["id"])
    match = f"{fx['teams']['home']['name']} vs {fx['teams']['away']['name']}"
    now = datetime.now(UTC).isoformat(timespec="seconds")
    n = 0

    def team(market: str, prob: float | None, mean: float | None = None,
             line: float | None = None) -> None:
        nonlocal n
        n += int(ft.log_prediction(
            fixture_id=fid, match=match, market=market, player_id=0, player_name="-",
            pred_mean=mean, pred_prob=prob, line=line, notes=f"as_of={now}",
        ))

    for side in ("home", "draw", "away"):
        team(f"team_1x2_{side}", markets["1x2"][side])
    team("team_btts", markets["btts"]["yes"])
    team("team_goals_ou_2.5", markets["goles_ou"][2.5]["over"],
         pred.get("means", {}).get("goals_h", 0) + pred.get("means", {}).get("goals_a", 0), 2.5)
    team("team_corners_ou_9.5", markets["corners_ou"][9.5]["over"], line=9.5)
    team("team_yellows_ou_3.5", markets["tarjetas_ou"][3.5]["over"], line=3.5)
    if ko is not None:
        for side in ("home", "away"):
            team(f"team_se_clasifica_{side}", ko["se_clasifica"][side])

    for props in (props_h, props_a):
        for _, r in props.head(5).iterrows():
            n += int(ft.log_prediction(
                fixture_id=fid, match=match, market="shots",
                player_id=int(r["player_id"]), player_name=str(r["player_name"]),
                pred_mean=float(r.get("mu_shots", 0)), notes=f"as_of={now}",
            ))
            if r.get("p_scores") is not None:
                n += int(ft.log_prediction(
                    fixture_id=fid, match=match, market="anota",
                    player_id=int(r["player_id"]), player_name=str(r["player_name"]),
                    pred_prob=float(r["p_scores"]), notes=f"as_of={now}",
                ))
    return n


def cmd_pre_day(date_str: str | None) -> None:
    settings = get_settings()
    key = settings.api_football_key
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    fixtures = [f for f in _day_fixtures(key, date_str)
                if f["fixture"]["status"]["short"] == "NS"]
    if not fixtures:
        print(f"Sin partidos NS del Mundial el {date_str}.")
        return
    engine = _build_engine()
    table = _player_table()

    lines = [f"# Mundial — predicciones del {date_str}", ""]
    total_logged = 0
    for fx in fixtures:
        home, away = fx["teams"]["home"]["name"], fx["teams"]["away"]["name"]
        when = pd.Timestamp(fx["fixture"]["date"][:10])
        ko_match = _is_knockout(fx)
        pred = engine.predict_match(home, away, when=when)
        pmfs, means = pred["pmfs"], pred["means"]
        markets = proj.project_all(pmfs)
        ko = proj.knockout_markets(pmfs) if ko_match else None
        p_et = ko["p_prorroga"] if ko else 0.0
        te_factor = 1.0 + p_et * (30.0 / 90.0) * proj.ET_FATIGUE if ko else 1.0
        horizon = "120" if ko_match else "90"
        props_h = _props_for(table, home, means, "h", horizon=horizon, te_factor=te_factor)
        props_a = _props_for(table, away, means, "a", horizon=horizon, te_factor=te_factor)

        lines += [f"## {home} vs {away} — {fx['league'].get('round', '')}", ""]
        lines += [f"Cantidades (90'): goles {means['goals_h']:.2f}-{means['goals_a']:.2f} · "
                  f"córners {means['corners_h']:.1f}-{means['corners_a']:.1f} · "
                  f"amarillas {means['yellows_h']:.1f}-{means['yellows_a']:.1f} · "
                  f"remates {means['shots_h']:.1f}-{means['shots_a']:.1f} · "
                  f"al arco {means['sot_h']:.1f}-{means['sot_a']:.1f}", ""]
        m1 = markets["1x2"]
        lines += [f"1X2: {m1['home']:.0%} / {m1['draw']:.0%} / {m1['away']:.0%} · "
                  f"BTTS {markets['btts']['yes']:.0%} · "
                  f"O2.5 {markets['goles_ou'][2.5]['over']:.0%} · "
                  f"O9.5 córners {markets['corners_ou'][9.5]['over']:.0%} · "
                  f"O3.5 amarillas {markets['tarjetas_ou'][3.5]['over']:.0%}", ""]
        if ko:
            q = ko["se_clasifica"]
            mv = ko["metodo_victoria"]
            lines += [f"**Eliminatoria** (P prórroga {ko['p_prorroga']:.0%}): "
                      f"se clasifica {home} {q['home']:.0%} / {away} {q['away']:.0%} · "
                      f"método: 90' {mv['home_90']:.0%}/{mv['away_90']:.0%}, "
                      f"ET {mv['home_et']:.0%}/{mv['away_et']:.0%}, "
                      f"pens {mv['home_pens']:.0%}/{mv['away_pens']:.0%}", ""]
            g_te = ko["te"]["goles"][2.5]["over"]
            lines += [f"Totales TE: O2.5 goles {g_te:.0%} · "
                      f"O9.5 córners {ko['te']['corners'][9.5]['over']:.0%} · "
                      f"O3.5 amarillas {ko['te']['tarjetas'][3.5]['over']:.0%}", ""]
        for name, props in ((home, props_h), (away, props_a)):
            if props.empty:
                lines += [f"### {name}: sin datos de jugadores", ""]
                continue
            lines += [f"### Props {name} (XI probable, horizonte {horizon}')", ""]
            lines += _fmt_props(props) + [""]
        total_logged += _log_match_predictions(fx, pred, markets, ko, props_h, props_a)

    REPORTS_DAILY.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DAILY / f"{date_str}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte -> {out}")
    print(f"Predicciones registradas en el forward-test: {total_logged}")


def cmd_pre_kickoff(fixture_id: int, *, force_window: bool = False) -> None:
    settings = get_settings()
    key = settings.api_football_key
    d = _get_cached(key, "/fixtures", {"id": fixture_id}, f"fx_{fixture_id}")
    resp = d.get("response", [])
    if not resp:
        print(f"Fixture {fixture_id} no encontrado.")
        return
    fx = resp[0]
    kickoff = pd.Timestamp(fx["fixture"]["date"]).tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    mins = (kickoff - now).total_seconds() / 60.0
    if not force_window and not (0 <= mins <= LINEUP_WINDOW_MIN):
        print(f"Fuera de la ventana de lineups ({mins:.0f} min al kickoff; "
              f"ventana -{LINEUP_WINDOW_MIN}→0). Salgo sin llamar a la API.")
        return
    lu = _get_cached(key, "/fixtures/lineups", {"fixture": fixture_id},
                     f"lineups_{fixture_id}")
    teams_lu = lu.get("response", [])
    if len(teams_lu) < 2:
        print("Lineups todavía no publicados.")
        return

    engine = _build_engine()
    table = _player_table()
    home, away = fx["teams"]["home"]["name"], fx["teams"]["away"]["name"]
    when = pd.Timestamp(fx["fixture"]["date"][:10])
    pred = engine.predict_match(home, away, when=when)
    ko_match = _is_knockout(fx)
    horizon = "120" if ko_match else "90"
    fid = int(fx["fixture"]["id"])

    for side, name in (("h", home), ("a", away)):
        xi = {int(p["player"]["id"]) for t in teams_lu
              if t["team"]["name"] == name for p in t.get("startXI", [])}
        before = _props_for(table, name, pred["means"], side, horizon=horizon)
        after = _props_for(table, name, pred["means"], side, horizon=horizon, lineup=xi or None)
        if before.empty or after.empty:
            continue
        print(f"\n=== {name} — delta con XI confirmado ===")
        merged = before.merge(after, on="player_id", suffixes=("_ant", "_conf"))
        merged["delta"] = (merged.get("mu_shots_conf", 0) - merged.get("mu_shots_ant", 0)).abs()
        for _, r in merged.sort_values("delta", ascending=False).head(5).iterrows():
            print(f"  {r['player_name_ant']}: μ remates "
                  f"{r.get('mu_shots_ant', 0):.2f} → {r.get('mu_shots_conf', 0):.2f}")
        for _, r in after.head(5).iterrows():
            ft.log_prediction(
                fixture_id=fid, match=f"{home} vs {away}", market="sot",
                player_id=int(r["player_id"]), player_name=str(r["player_name"]),
                pred_mean=float(r.get("mu_sot", 0)), notes="xi_confirmado",
            )


def cmd_post_day(date_str: str | None) -> None:
    settings = get_settings()
    key = settings.api_football_key
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    fixtures = [f for f in _day_fixtures(key, date_str, force=True)
                if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")]
    if not fixtures:
        print(f"Sin partidos terminados el {date_str}.")
        return
    from mundial_bot.collectors.nt_data import fetch_fixture_details, fixture_to_row
    fetch_fixture_details(key, [int(f["fixture"]["id"]) for f in fixtures])
    total_p = total_t = 0
    for fx in fixtures:
        fid = int(fx["fixture"]["id"])
        n_players = ft.settle_fixture(key, fid)
        detail = json.loads((NT_CACHE / f"fixture_{fid}.json").read_text(encoding="utf-8"))
        row = fixture_to_row(detail)
        actuals = {}
        if row is not None:
            gh, ga = int(row["home_score"]), int(row["away_score"])
            winner_h = (fx.get("teams", {}).get("home", {}) or {}).get("winner")
            actuals = {
                "goals_total": gh + ga,
                "corners_total": int(row["corners_h"]) + int(row["corners_a"]),
                "yellows_total": int(row["yellows_h"]) + int(row["yellows_a"]),
                "shots_total": int(row["shots_h"]) + int(row["shots_a"]),
                "sot_total": int(row["sot_h"]) + int(row["sot_a"]),
                "result": "home" if gh > ga else ("away" if ga > gh else "draw"),
                "btts": 1.0 if (gh > 0 and ga > 0) else 0.0,
                # ganador del fixture (incluye ET/pens si los hubo) = el que avanza
                "advanced": ("home" if winner_h is True
                             else ("away" if winner_h is False else None)),
            }
        n_team = ft.settle_team_fixture(fid, actuals)
        total_p += n_players
        total_t += n_team
        print(f"  {fx['teams']['home']['name']} vs {fx['teams']['away']['name']}: "
              f"{n_players} props + {n_team} mercados liquidados")
    print(f"\nLiquidado: {total_p} props, {total_t} mercados de equipo.")
    print("Nota: la tabla de jugadores/equipos se refresca sola en el próximo pre-day "
          "(los fixtures nuevos ya quedaron cacheados).")


def cmd_weekly() -> None:
    s = ft.summary()
    lines = ["# Forward-test — reporte semanal", "",
             f"Predicciones: {s['total']} (liquidadas {s['settled']}, "
             f"pendientes {s['pending']})",
             f"Brier medio: {s['brier_mean']:.4f}" if s["brier_mean"] is not None
             else "Brier medio: sin liquidadas aún",
             f"MAE de medias: {s['mae_mean']:.3f}" if s["mae_mean"] is not None
             else "MAE: sin liquidadas aún", ""]
    with ft._connect() as conn:  # noqa: SLF001 — reporte interno del mismo paquete
        rows = conn.execute(
            """SELECT market, COUNT(*) n, AVG(brier) brier
               FROM props_log WHERE settled_at IS NOT NULL GROUP BY market
               ORDER BY n DESC"""
        ).fetchall()
    if rows:
        lines += ["| Mercado | n | Brier |", "|---|---|---|"]
        lines += [f"| {m} | {n} | {b:.4f} |" if b is not None else f"| {m} | {n} | — |"
                  for m, n, b in rows]
    out = REPORTS_DAILY.parent / "weekly_forward_test.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReporte -> {out}")


def cmd_add_odds(fixture_id: int, player_id: int, market: str,
                 line: float | None, odds: float) -> None:
    with ft._connect() as conn:  # noqa: SLF001
        cur = conn.execute(
            """UPDATE props_log SET line=COALESCE(?, line), odds=?, book='bet365'
               WHERE fixture_id=? AND player_id=? AND market=?""",
            (line, odds, fixture_id, player_id, market),
        )
    if cur.rowcount:
        print(f"OK: cuota {odds} cargada en {market} (fixture {fixture_id}).")
    else:
        print("No encontré esa predicción (fixture/player/market). Nada cambiado.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m mundial_bot.wc.daily", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d1 = sub.add_parser("pre-day", help="reporte + registro de predicciones del día")
    d1.add_argument("--date", default=None)
    d2 = sub.add_parser("pre-kickoff", help="delta de props con XI confirmado")
    d2.add_argument("--fixture", type=int, required=True)
    d2.add_argument("--force-window", action="store_true",
                    help="saltear el chequeo de ventana (para pruebas)")
    d3 = sub.add_parser("post-day", help="liquida contra resultados reales")
    d3.add_argument("--date", default=None)
    sub.add_parser("weekly", help="reporte acumulado del forward-test")
    d5 = sub.add_parser("add-odds", help="carga manual de línea/cuota bet365")
    d5.add_argument("--fixture", type=int, required=True)
    d5.add_argument("--player", type=int, default=0)
    d5.add_argument("--market", required=True)
    d5.add_argument("--line", type=float, default=None)
    d5.add_argument("--odds", type=float, required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "pre-day":
            cmd_pre_day(a.date)
        elif a.cmd == "pre-kickoff":
            cmd_pre_kickoff(a.fixture, force_window=a.force_window)
        elif a.cmd == "post-day":
            cmd_post_day(a.date)
        elif a.cmd == "weekly":
            cmd_weekly()
        elif a.cmd == "add-odds":
            cmd_add_odds(a.fixture, a.player, a.market, a.line, a.odds)
    except requests.RequestException as exc:
        print(f"Error de red con API-Football: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
