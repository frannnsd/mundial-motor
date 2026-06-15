"""Validador de calibración de los modelos de conteo (córners/tarjetas).

Walk-forward (entrena con el pasado, predice el presente): compara el TOTAL esperado
contra el real y mide la calibración de P(over) a varias líneas. Reporta el sesgo y el
factor de corrección que mejor matchea la frecuencia real de over — el mismo criterio
que usa la auto-calibración de CornersModel. Correr cuando entran datos nuevos para
revisar si el factor sigue siendo el adecuado:

    python scripts/diagnose_bias.py
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from mundial_bot.backtest.count_backtest import _per_match_table  # noqa: E402
from mundial_bot.collectors.team_stats import load_team_stats  # noqa: E402
from mundial_bot.models.cards_model import CardsModel  # noqa: E402
from mundial_bot.models.corners_model import CornersModel  # noqa: E402
from mundial_bot.models.count_market import over_under  # noqa: E402

LINES = (7.5, 8.5, 9.5, 10.5, 11.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


def collect(market="córners"):
    df = load_team_stats()
    matches = _per_match_table(df)
    start = int(len(matches) * 0.4)
    col = "corners_total" if market == "córners" else "cards_total"
    rows = []
    for i in range(start, len(matches)):
        m = matches.iloc[i]
        train = df[df["date"] < m["date"]]
        train_m = matches[matches["date"] < m["date"]]
        if len(train_m) < 30:
            continue
        if market == "córners":
            model = CornersModel.from_events(train)
            pred = model.predict(m["home"], m["away"])
        else:
            model = CardsModel.from_events(train)
            pred = model.predict(m["home"], m["away"], referee=m["referee"])
        rows.append((pred.total, getattr(model, "dispersion", 1.0), m[col]))
    return np.array(rows)  # cols: pred_total, dispersion, actual


def calib(rows, mean_factor, disp_factor, lines):
    preds = rows[:, 0] * mean_factor
    disps = rows[:, 1] * disp_factor
    actuals = rows[:, 2]
    err = 0.0
    out = []
    for line in lines:
        p_over = np.array([
            over_under(pt, line, variance=pt * dp)[0]
            for pt, dp in zip(preds, disps, strict=True)
        ])
        real = (actuals > line).mean()
        out.append((line, p_over.mean(), real))
        err += (p_over.mean() - real) ** 2
    return out, err


def report(market, lines):
    rows = collect(market)
    print(f"\n######## {market.upper()} ########")
    print(f"n={len(rows)} · media pred {rows[:,0].mean():.2f} · real {rows[:,2].mean():.2f} "
          f"· sesgo {rows[:,0].mean()-rows[:,2].mean():+.2f}")
    _, err0 = calib(rows, 1.0, 1.0, lines)
    print(f"error² sin corrección = {err0:.4f}")
    # Mejor factor de media (la dispersión no ayuda, ya visto)
    best = None
    for mf in np.linspace(0.90, 1.25, 36):
        _, err = calib(rows, mf, 1.0, lines)
        if best is None or err < best[1]:
            best = (mf, err)
    print(f"MEJOR factor de media = {best[0]:.3f} · error² = {best[1]:.4f}")
    out, _ = calib(rows, best[0], 1.0, lines)
    for line, pm, real in out:
        print(f"   línea {line}: corregido {pm:.1%} · real {real:.1%}  ({pm-real:+.1%})")


def main():
    report("córners", LINES)
    report("tarjetas", CARD_LINES)


if __name__ == "__main__":
    main()
