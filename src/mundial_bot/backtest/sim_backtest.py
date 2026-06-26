"""Backtest del modelo contra los partidos del Mundial 2026 ya jugados.

Entrena el Dixon-Coles SIN los resultados del Mundial (para no hacer trampa) y
predice cada partido jugado. La calibración del ritmo de gol es **walk-forward**:
para cada partido, el factor se calcula SOLO con los partidos anteriores (como
pasaría en vivo), así el número que mostramos es honesto (no in-sample).

Devuelve, por mercado (1X2, Más/Menos 2.5, ambos marcan): Brier + acierto, y una
curva de calibración del Over 2.5.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from mundial_bot.collectors.wc_results import load_wc_results
from mundial_bot.pipeline import build_models

_CAL_MIN_PRIOR = 12          # partidos previos mínimos para empezar a calibrar
_CAL_BOUNDS = (0.9, 1.3)
_GRID = 14


def _brier_multi(probs: list[float], idx: int) -> float:
    target = [0.0, 0.0, 0.0]
    target[idx] = 1.0
    return sum((probs[i] - target[i]) ** 2 for i in range(3))


def _calibration(pairs: list[tuple[float, bool]], bins: int = 5) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sub = [(p, o) for p, o in pairs if (p >= lo and (p < hi or i == bins - 1))]
        if sub:
            out.append({
                "bucket": f"{int(lo * 100)}-{int(hi * 100)}%",
                "n": len(sub),
                "predicted": round(float(np.mean([p for p, _ in sub])), 3),
                "actual": round(float(np.mean([int(o) for _, o in sub])), 3),
            })
        else:
            out.append({"bucket": f"{int(lo * 100)}-{int(hi * 100)}%", "n": 0,
                        "predicted": None, "actual": None})
    return out


def _markets_from_xg(lh: float, la: float) -> tuple[list[float], float, float]:
    """1X2, P(over 2.5), P(ambos marcan) desde xG (Poisson)."""
    k = np.arange(_GRID)
    pm = np.outer(poisson.pmf(k, max(lh, 1e-9)), poisson.pmf(k, max(la, 1e-9)))
    pm = pm / pm.sum()
    i = k.reshape(-1, 1)
    j = k.reshape(1, -1)
    margin = i - j
    total = i + j
    p = [float(pm[margin > 0].sum()), float(pm[margin == 0].sum()), float(pm[margin < 0].sum())]
    over = float(pm[total > 2.5].sum())
    btts = float(pm[(i >= 1) & (j >= 1)].sum())
    return p, over, btts


def run_backtest() -> dict:
    models = build_models(fit_goals=True)  # SIN extra_results del Mundial → sin leakage
    goals = models.goals
    df = load_wc_results()
    if goals is None or df.empty:
        return {"n": 0, "markets": {}, "calibration_over_2_5": [], "sample": [], "calibration": 1.0}

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    goals.calibration = 1.0  # las predicciones crudas; calibramos walk-forward acá

    # xG crudo de cada partido jugado (una sola pasada).
    games: list[dict] = []
    for _, r in df.iterrows():
        h, a = str(r["home_team"]), str(r["away_team"])
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None or not goals.can_predict(h, a):
            continue
        try:
            _, hx, ax = goals.score_matrix(h, a, neutral=True)
        except Exception:  # noqa: BLE001
            continue
        games.append({"home": h, "away": a, "hx": float(hx), "ax": float(ax),
                      "hs": int(hs), "as": int(as_)})

    # Walk-forward: el factor de cada partido sale SOLO de los anteriores.
    rows = []
    cum_act = cum_pred = 0.0
    nprior = 0
    last_factor = 1.0
    for g in games:
        if nprior >= _CAL_MIN_PRIOR and cum_pred > 0:
            last_factor = min(max(cum_act / cum_pred, _CAL_BOUNDS[0]), _CAL_BOUNDS[1])
        p, over, btts = _markets_from_xg(g["hx"] * last_factor, g["ax"] * last_factor)
        hs, as_ = g["hs"], g["as"]
        res_idx = 0 if hs > as_ else (1 if hs == as_ else 2)
        rows.append({
            "home": g["home"], "away": g["away"], "score": f"{hs}-{as_}",
            "p": p, "res_idx": res_idx,
            "p_over": over, "over": (hs + as_) > 2.5,
            "p_btts": btts, "btts": hs > 0 and as_ > 0,
        })
        cum_act += hs + as_
        cum_pred += g["hx"] + g["ax"]
        nprior += 1

    n = len(rows)
    if n == 0:
        return {"n": 0, "markets": {}, "calibration_over_2_5": [], "sample": [], "calibration": 1.0}

    brier_1x2 = float(np.mean([_brier_multi(x["p"], x["res_idx"]) for x in rows]))
    acc_1x2 = float(np.mean([int(np.argmax(x["p"]) == x["res_idx"]) for x in rows]))
    brier_ou = float(np.mean([(x["p_over"] - int(x["over"])) ** 2 for x in rows]))
    acc_ou = float(np.mean([int((x["p_over"] >= 0.5) == x["over"]) for x in rows]))
    brier_btts = float(np.mean([(x["p_btts"] - int(x["btts"])) ** 2 for x in rows]))
    acc_btts = float(np.mean([int((x["p_btts"] >= 0.5) == x["btts"]) for x in rows]))

    labels = ["1", "X", "2"]
    sample = [{
        "match": f'{x["home"]} vs {x["away"]}',
        "score": x["score"],
        "p_home": round(x["p"][0], 3), "p_draw": round(x["p"][1], 3), "p_away": round(x["p"][2], 3),
        "pred": labels[int(np.argmax(x["p"]))],
        "hit": int(np.argmax(x["p"]) == x["res_idx"]),
    } for x in rows]

    return {
        "n": n,
        "markets": {
            "1x2": {"brier": round(brier_1x2, 4), "accuracy": round(acc_1x2, 4)},
            "over_2_5": {"brier": round(brier_ou, 4), "accuracy": round(acc_ou, 4)},
            "btts": {"brier": round(brier_btts, 4), "accuracy": round(acc_btts, 4)},
        },
        "calibration_over_2_5": _calibration([(x["p_over"], x["over"]) for x in rows]),
        "sample": sample,
        "calibration": round(last_factor, 3),
    }
