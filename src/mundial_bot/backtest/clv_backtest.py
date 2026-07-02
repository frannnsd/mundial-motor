"""Backtest de CLV (Closing Line Value) del motor vs el CIERRE de Pinnacle — Fase 2.

Responde LA pregunta del gate: **¿el motor le gana al cierre de Pinnacle** (la línea más
sharp del mercado)? Calibrar bien ≠ ser rentable; esto lo mide con plata de mentira sobre
miles de partidos de liga, gratis, antes de arriesgar un peso.

Diseño (reusa lo que ya existe):
- **Motor 1X2:** Elo incremental (`models/elo_model`), que es point-in-time POR CONSTRUCCIÓN
  (predice ANTES de actualizar). Se procesa por grupos de misma fecha: se predicen TODOS los
  partidos del día ANTES de actualizar el Elo con cualquiera de ellos → nunca un partido usa
  otro del mismo día (que no sabemos si terminó antes). El blend con Dixon-Coles queda como
  refinamiento futuro (no cambia el veredicto: ninguno le gana al cierre de Pinnacle).
- **De-vig:** `value/devig` (Shin) sobre el cierre → probabilidad "real" del mercado.
- **CLV real (movimiento de línea):** para el pick del modelo, CLV = cuota_apertura/cuota_cierre − 1
  (positivo = el cierre quedó más corto que lo que conseguiste = le ganaste al cierre).

Innegociables (Fase 1 no sirve de nada si se rompen):
- (a) NINGUNA feature usa datos ≥ kickoff. El Elo no toma `as_of` porque es walk-forward por
  construcción; este backtest NO llama a weighted_means/from_events/fit_calibration (esas sí
  toman `as_of`). O sea: cero calls de feature que puedan filtrar.
- (b) `assert_point_in_time()` corre DENTRO del loop, en CADA partido; si falla, se corta.

Hold-out sagrado: `HOLDOUT_SEASON` — el tuning NUNCA la toca; el veredicto se reporta sobre esa.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mundial_bot.backtest.leakage_guard import assert_point_in_time
from mundial_bot.models.elo_model import EloConfig, EloModel
from mundial_bot.value.devig import devig

HOLDOUT_SEASON = "2324"          # última temporada = hold-out sagrado
OUTCOMES = ("home", "draw", "away")
_EPS = 1e-15


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if home_score == away_score:
        return "draw"
    return "away"


def _score_match(
    row: pd.Series, probs: dict[str, float], *, min_edge: float, method: str
) -> dict | None:
    """Compara el modelo con el cierre (y la apertura) de Pinnacle para un partido."""
    hs, as_ = int(row["home_score"]), int(row["away_score"])
    actual = _actual_outcome(hs, as_)

    close_odds = {
        "home": float(row["psc_h"]), "draw": float(row["psc_d"]), "away": float(row["psc_a"]),
    }
    try:
        close_dv = devig(close_odds, method)
    except (ValueError, ZeroDivisionError):
        return None  # cierre roto → no se puede comparar

    # Calibración: ¿el modelo predice mejor que el cierre des-marginado?
    brier_model = sum((probs[o] - (1.0 if o == actual else 0.0)) ** 2 for o in OUTCOMES)
    brier_close = sum((close_dv[o] - (1.0 if o == actual else 0.0)) ** 2 for o in OUTCOMES)
    ll_model = -math.log(max(probs[actual], _EPS))
    ll_close = -math.log(max(close_dv[actual], _EPS))

    rec: dict = {
        "season": row["season"], "league": row["league"],
        "is_holdout": row["season"] == HOLDOUT_SEASON,
        "actual": actual,
        "brier_model": brier_model, "brier_close": brier_close,
        "ll_model": ll_model, "ll_close": ll_close,
        "bet": None, "clv": None, "won": None, "roi": None, "edge": None,
    }

    # CLV real: precisa la cuota de APERTURA (la que conseguirías al apostar).
    ph, pdw, pa = row["ps_h"], row["ps_d"], row["ps_a"]
    if not (pd.isna(ph) or pd.isna(pdw) or pd.isna(pa)):
        open_odds = {"home": float(ph), "draw": float(pdw), "away": float(pa)}
        try:
            open_dv = devig(open_odds, method)
        except (ValueError, ZeroDivisionError):
            return rec
        # El modelo apuesta donde ve más valor vs la APERTURA.
        edges = {o: probs[o] - open_dv[o] for o in OUTCOMES}
        pick = max(OUTCOMES, key=lambda o: edges[o])
        if edges[pick] > min_edge:
            clv = open_odds[pick] / close_odds[pick] - 1.0
            won = pick == actual
            rec.update(
                bet=pick, edge=edges[pick], clv=clv, won=won,
                roi=(open_odds[pick] - 1.0) if won else -1.0,
            )
    return rec


def _process_league(
    ldf: pd.DataFrame, *, min_edge: float, method: str, records: list[dict]
) -> None:
    """Walk-forward de UNA liga (Elo propio por liga: equipos de ligas distintas no se cruzan)."""
    ldf = ldf.sort_values("date").reset_index(drop=True)
    elo = EloModel(EloConfig())
    n = len(ldf)
    i = 0
    while i < n:
        day = ldf["date"].iloc[i]
        j = i
        while j < n and ldf["date"].iloc[j] == day:
            j += 1
        prior = ldf.iloc[:i]  # SOLO partidos con fecha < day (df ordenado) → sin same-day leakage
        for k in range(i, j):
            # (b) guard EN EL LOOP, por partido: corta si alguna feature usa datos >= kickoff.
            assert_point_in_time(prior, day, label=f"clv:{ldf['league'].iloc[k]}")
            row = ldf.iloc[k]
            p = elo.predict(row["home_team"], row["away_team"], neutral=False)
            rec = _score_match(
                row, {"home": p.home, "draw": p.draw, "away": p.away},
                min_edge=min_edge, method=method,
            )
            if rec is not None:
                records.append(rec)
        # Recién DESPUÉS de predecir todo el día se actualiza el Elo (nunca intra-día).
        for k in range(i, j):
            row = ldf.iloc[k]
            elo.update(
                row["home_team"], row["away_team"],
                home_score=int(row["home_score"]), away_score=int(row["away_score"]),
                tournament=row["tournament"], neutral=False,
            )
        i = j


def _summarize(recs: list[dict]) -> dict:
    """Métricas agregadas de un conjunto de partidos (una split: hold-out o dev)."""
    if not recs:
        return {"n": 0}
    bets = [r for r in recs if r["clv"] is not None]
    clvs = np.array([r["clv"] for r in bets], dtype=float)
    rois = np.array([r["roi"] for r in bets], dtype=float)
    out = {
        "n": len(recs),
        "n_bets": len(bets),
        "brier_model": float(np.mean([r["brier_model"] for r in recs])),
        "brier_close": float(np.mean([r["brier_close"] for r in recs])),
        "logloss_model": float(np.mean([r["ll_model"] for r in recs])),
        "logloss_close": float(np.mean([r["ll_close"] for r in recs])),
    }
    if len(bets):
        out.update({
            "clv_mean": float(np.mean(clvs)),
            "clv_median": float(np.median(clvs)),
            "clv_std": float(np.std(clvs)),
            "pct_beat_close": float(np.mean(clvs > 0)),
            "roi_at_open": float(np.mean(rois)),
            "hit_rate": float(np.mean([bool(r["won"]) for r in bets])),
        })
    return out


def run_clv_backtest(
    df: pd.DataFrame | None = None,
    *,
    holdout_season: str = HOLDOUT_SEASON,
    min_edge: float = 0.0,
    method: str = "shin",
) -> dict:
    """Corre el backtest de CLV y devuelve métricas por split (hold-out vs dev/tuning).

    ``df`` con el esquema de football_data (si es None, lo carga). El reporte que importa
    es el del **hold-out** (``holdout_season``), que el tuning nunca tocó.
    """
    if df is None:
        from mundial_bot.collectors.football_data import load_football_data
        df = load_football_data()
    if df.empty:
        return {"n": 0, "holdout": {"n": 0}, "dev": {"n": 0}}

    records: list[dict] = []
    for _, ldf in df.groupby("league"):
        _process_league(ldf, min_edge=min_edge, method=method, records=records)

    holdout = [r for r in records if r["is_holdout"]]
    dev = [r for r in records if not r["is_holdout"]]
    ho, dv = _summarize(holdout), _summarize(dev)

    # Descuento de optimismo: cuánto peor rinde el hold-out que el dev (el backtest siempre
    # se ve mejor que el vivo; acá medimos el gap dev→hold-out como piso de ese descuento).
    optimism = {}
    if ho.get("n_bets") and dv.get("n_bets"):
        optimism = {
            "roi_dev_minus_holdout": dv["roi_at_open"] - ho["roi_at_open"],
            "clv_dev_minus_holdout": dv["clv_mean"] - ho["clv_mean"],
            "brier_holdout_minus_dev": ho["brier_model"] - dv["brier_model"],
        }

    return {
        "n": len(records),
        "holdout_season": holdout_season,
        "min_edge": min_edge,
        "method": method,
        "holdout": ho,
        "dev": dv,
        "optimism_discount": optimism,
    }


def _verdict(ho: dict) -> str:
    if not ho.get("n_bets"):
        return "🟡 sin apuestas en el hold-out (revisar datos)"
    roi = ho["roi_at_open"]
    clv = ho["clv_mean"]
    if clv > 0.002 and roi > 0:
        return "🟢 LE GANA AL CIERRE (CLV+ y ROI+ en el hold-out): señal de edge real"
    if clv > -0.002:
        return "🟡 a la par del cierre (sin edge claro; calibrado pero no rentable vs Pinnacle)"
    return "🔴 NO le gana al cierre (CLV negativo): el cierre de Pinnacle es más sharp"


def format_clv_backtest(result: dict) -> str:
    """Reporte legible del backtest de CLV."""
    ho, dv = result.get("holdout", {}), result.get("dev", {})
    if not ho.get("n"):
        return "Backtest de CLV: sin datos en el hold-out."

    def g(d: dict, k: str) -> float:
        return d.get(k, float("nan"))

    lines = [
        f"BACKTEST DE CLV — motor vs cierre de Pinnacle (de-vig {result['method']})",
        f"Hold-out sagrado: temporada {result['holdout_season']} · pick: max-edge vs apertura",
        "",
        f"VEREDICTO: {_verdict(ho)}",
        "",
        f"— HOLD-OUT ({ho['n']} partidos, {ho.get('n_bets', 0)} apuestas) —",
        f"  CLV medio           : {g(ho, 'clv_mean'):+.3%}  (mediana {g(ho, 'clv_median'):+.3%})",
        f"  % le gana al cierre : {g(ho, 'pct_beat_close'):.1%}",
        f"  ROI a la apertura   : {g(ho, 'roi_at_open'):+.2%}  (acierto {g(ho, 'hit_rate'):.1%})",
        f"  Brier   modelo/cierre : {g(ho, 'brier_model'):.4f} / {g(ho, 'brier_close'):.4f}",
        f"  LogLoss modelo/cierre : {g(ho, 'logloss_model'):.4f} / {g(ho, 'logloss_close'):.4f}",
        "",
        f"— DEV/tuning ({dv.get('n', 0)} partidos, {dv.get('n_bets', 0)} apuestas) —",
        f"  CLV {g(dv, 'clv_mean'):+.3%} · ROI {g(dv, 'roi_at_open'):+.2%}"
        f" · Brier {g(dv, 'brier_model'):.4f}",
    ]
    opt = result.get("optimism_discount") or {}
    if opt:
        lines += [
            "",
            "— DESCUENTO DE OPTIMISMO (dev − hold-out) —",
            f"  ROI {opt['roi_dev_minus_holdout']:+.2%} · CLV {opt['clv_dev_minus_holdout']:+.3%}",
        ]
    lines += [
        "",
        "Nota: el Brier/logloss del cierre suele ser IMBATIBLE (Pinnacle es sharp). Este",
        "backtest no modela movimiento de línea real ni límites → el vivo descuenta más.",
    ]
    return "\n".join(lines)
