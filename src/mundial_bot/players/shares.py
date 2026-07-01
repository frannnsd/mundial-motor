"""Shares por jugador: qué fracción de cada conteo del equipo produce cada uno.

Entrada: la tabla (fixture, equipo, jugador) de `collectors.players_wc`.
Salida: por jugador del equipo, la tasa por-90 de cada stat (con shrinkage) y su
share = fracción del total por-90 del equipo (ponderado por minutos jugados).

SHRINKAGE HACIA EL PUESTO (jugadores con pocos minutos):

    rate_shrunk = (min_jugados × rate_raw + K × rate_puesto) / (min_jugados + K)

con K = 180 minutos (≈ 2 partidos). Como `min_jugados × rate_raw = 90 × total`,
la fórmula es estable incluso con 0 minutos (cae al promedio del puesto). El
promedio del puesto (G/D/M/F) se calcula sobre TODOS los equipos del torneo —
mucha más muestra que un plantel solo.

Los shares de acá son descriptivos (ponderados por minutos históricos); la
normalización final "suma 1 sobre el XI esperado" la hace `props.match_props`
después de seleccionar los 11 (propiedad de coherencia con el total del equipo).
"""

from __future__ import annotations

import pandas as pd

# Stats que se repartem entre jugadores (red se excluye: demasiado rara para shares).
SHARE_STATS = (
    "shots", "sot", "goals", "assists", "fouls_committed", "fouls_drawn",
    "yellow", "tackles",
)
SHRINK_MINUTES_K = 180.0  # minutos de "fe" en el promedio del puesto (≈ 2 partidos)


def position_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Tasa por-90 de cada stat por PUESTO (G/D/M/F), sobre todo el torneo.

    Ponderada por minutos: rate = 90 × Σ stat / Σ minutos del puesto. Es el
    prior del shrinkage. Incluye una fila "" con el promedio global (fallback
    para posiciones desconocidas).
    """
    df = table.copy()
    df["position"] = df["position"].fillna("").astype(str)
    grouped = df.groupby("position")[["minutes", *SHARE_STATS]].sum()
    total = df[["minutes", *SHARE_STATS]].sum().to_frame().T
    total.index = pd.Index([""], name="position")
    grouped = pd.concat([grouped, total.loc[~total.index.isin(grouped.index)]])
    minutes = grouped["minutes"].clip(lower=1.0)
    rates = grouped[list(SHARE_STATS)].div(minutes, axis=0) * 90.0
    return rates


def player_shares(table: pd.DataFrame, team: str) -> pd.DataFrame:
    """Tasas por-90 (con shrinkage por puesto) y shares de los jugadores de `team`.

    Devuelve una fila por jugador con:
      - matches / minutes / min_avg / min_avg_starter / min_avg_sub / n_started:
        insumos de `props.expected_minutes` (min_avg promedia TODAS las citaciones,
        incluidas las de 0 minutos — es la esperanza de minutos con rotación).
      - rate_<stat>: tasa por-90 shrunk hacia el promedio del puesto.
      - share_<stat>: fracción del total por-90 del equipo, ponderada por minutos
        históricos (Σ share = 1 por stat con actividad).
    """
    pos_rates = position_rates(table)
    rows = table[table["team"] == team]
    if rows.empty:
        return pd.DataFrame()

    out: list[dict] = []
    for (pid, name), g in rows.groupby(["player_id", "player_name"], sort=False):
        minutes = float(g["minutes"].sum())
        position = _mode_position(g)
        started = g[(~g["substitute"].astype(bool)) & (g["minutes"] > 0)]
        subbed = g[(g["substitute"].astype(bool)) & (g["minutes"] > 0)]
        rec: dict = {
            "player_id": int(pid),
            "player_name": str(name),
            "position": position,
            "matches": int((g["minutes"] > 0).sum()),
            "minutes": minutes,
            "min_avg": float(g["minutes"].mean()),
            "min_avg_starter": float(started["minutes"].mean()) if len(started) else 0.0,
            "min_avg_sub": float(subbed["minutes"].mean()) if len(subbed) else 0.0,
            "n_started": int(len(started)),
        }
        prior = pos_rates.loc[position if position in pos_rates.index else ""]
        for stat in SHARE_STATS:
            total = float(g[stat].sum())
            # min × rate_raw = 90 × total → estable aun con 0 minutos.
            rec[f"rate_{stat}"] = (90.0 * total + SHRINK_MINUTES_K * float(prior[stat])) / (
                minutes + SHRINK_MINUTES_K
            )
        out.append(rec)

    df = pd.DataFrame(out)
    for stat in SHARE_STATS:
        weighted = df[f"rate_{stat}"] * df["minutes"]
        denom = float(weighted.sum())
        df[f"share_{stat}"] = weighted / denom if denom > 0 else 0.0
    return df.sort_values("minutes", ascending=False).reset_index(drop=True)


def _mode_position(g: pd.DataFrame) -> str:
    """Puesto más frecuente del jugador en la muestra ('' si no hay dato)."""
    pos = g["position"].fillna("").astype(str)
    pos = pos[pos != ""]
    return str(pos.mode().iloc[0]) if len(pos) else ""
