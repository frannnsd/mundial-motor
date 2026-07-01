"""Fase B — descarga real de stats por jugador + ejemplo de props del próximo partido.

1. Baja (con cache a disco, 1 llamada por fixture + delay) los /fixtures/players
   de TODOS los partidos FT del Mundial 2026 y consolida la tabla por jugador.
2. Toma el primer partido PRÓXIMO (status NS) del Mundial.
3. Calcula los shares de los dos equipos y reparte totales de equipo razonables
   en props por jugador (XI probable — partido futuro, sin lineups confirmados).
4. Escribe el reporte en reports/fase_b_ejemplo.md.

Uso:  .venv/Scripts/python.exe scripts/fase_b_ejemplo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from mundial_bot.collectors import players_wc
from mundial_bot.config import PROJECT_ROOT, get_settings
from mundial_bot.players import match_props, player_shares

REPORT_PATH = PROJECT_ROOT / "reports" / "fase_b_ejemplo.md"

# Totales de equipo razonables A MANO para el ejemplo (media, var por partido).
# El paso posterior es enchufar acá el cerebro unificado de la Fase A (research)
# — hoy esos cerebros están entrenados en clubes, no en selecciones, así que el
# ejemplo no los usa end-to-end. La capa de jugador NO recalcula totales.
TEAM_TOTALS = {
    "shots": (13.0, 16.0),
    "sot": (4.5, 5.0),
    "goals": (1.6, 1.7),
    "yellow": (2.2, 2.4),
    "fouls_committed": (11.0, 12.0),
}
TOP_N = 8


def _fmt_team_table(team: str, props) -> list[str]:
    """Tabla markdown con el top-N de jugadores del equipo por μ de remates."""
    top = props.sort_values("mu_shots", ascending=False).head(TOP_N)
    lines = [
        f"### {team}",
        "",
        "| Jugador | Pos | Min esp | μ remates | μ al arco | P(anota) | P(tarjeta) | μ faltas |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['player_name']} | {r['position']} | {r['exp_minutes']:.0f}' "
            f"| {r['mu_shots']:.2f} | {r['mu_sot']:.2f} | {r['p_scores']:.0%} "
            f"| {r['p_card']:.0%} | {r['mu_fouls_committed']:.2f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    settings = get_settings()
    if not settings.has_api_football:
        print("Falta API_FOOTBALL_KEY en .env — no puedo bajar los datos.")
        return 1
    key = settings.api_football_key

    # 1) Tabla consolidada de todos los partidos FT (cache JSON por fixture + CSV).
    table = players_wc.build_player_match_table(key)
    if table.empty:
        print("La tabla de jugadores vino vacía — revisar la API.")
        return 1
    print(f"Tabla: {len(table)} filas · {table['fixture_id'].nunique()} fixtures · "
          f"{table['team'].nunique()} equipos · {table['player_id'].nunique()} jugadores")

    # 2) Próximo partido del Mundial (NS, el primero por fecha).
    upcoming = players_wc.fetch_upcoming_fixtures(key)
    if not upcoming:
        print("No hay partidos NS en la lista — ¿terminó el torneo?")
        return 1
    fx = upcoming[0]
    print(f"Próximo partido: {fx['home']} vs {fx['away']} ({fx['date']}) "
          f"[fixture {fx['fixture_id']}]")

    # 3) Props por equipo — XI probable (partido futuro: SIN lineup confirmado,
    #    para no filtrar información que todavía no existe).
    out = [
        "# Fase B — Ejemplo real: props por jugador",
        "",
        f"**Partido:** {fx['home']} vs {fx['away']} — {fx['date']} "
        f"(fixture {fx['fixture_id']})",
        "",
        "XI **probable** (sin alineaciones confirmadas: es un partido futuro — la capa",
        "solo usa lineups publicados para predicciones posteriores a su publicación).",
        "",
        "Totales de equipo del ejemplo (a mano, mismos para ambos): "
        + ", ".join(f"{k} {v[0]:g}" for k, v in TEAM_TOTALS.items())
        + ". μ por jugador = share × total × (min esp / 90); Σ jugadores == total.",
        "",
        f"Base: {len(table)} filas jugador-partido de "
        f"{table['fixture_id'].nunique()} partidos FT del Mundial 2026.",
        "",
    ]
    for team in (fx["home"], fx["away"]):
        shares = player_shares(table, team)
        if shares.empty:
            out += [f"### {team}", "", "Sin datos de jugadores en los partidos FT.", ""]
            continue
        props = match_props(TEAM_TOTALS, shares, lineup_confirmed=None)
        out += _fmt_team_table(team, props)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text("\n".join(out), encoding="utf-8")
    print(f"Reporte escrito en {REPORT_PATH}")
    print(f"Llamadas reales a la API en esta corrida: {players_wc.api_calls_made()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
