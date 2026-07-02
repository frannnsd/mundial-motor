"""Corre la competencia de cerebros (Fase A) y escribe el reporte.

Uso:  python scripts/run_brain_competition.py
Salida: reports/brain_competition_A.json (crudo) + tablas por stdout.
Determinístico (sin RNG); la config queda logueada en el JSON y el reporte.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from mundial_bot.collectors.football_data import load_football_stats
from mundial_bot.research.brains import BRAINS
from mundial_bot.research.competition import (
    QUANTITIES,
    evaluate_unified_holdout,
    run_competition,
    unify,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    t0 = time.time()
    df = load_football_stats()
    print(f"Partidos: {len(df)}  ({df['date'].min().date()} -> {df['date'].max().date()})")

    res = run_competition(df)
    print(f"Validacion: {res.n_scored_validation} | Hold-out: {res.n_scored_holdout} "
          f"| {time.time() - t0:.0f}s")

    weights = unify(res.validation)
    unified = evaluate_unified_holdout(res, weights)

    payload = {
        "config": asdict(res.config),
        "n_validation": res.n_scored_validation,
        "n_holdout": res.n_scored_holdout,
        "validation": {q: res.validation[q] for q in QUANTITIES if q in res.validation},
        "holdout_individual": {q: res.holdout[q] for q in QUANTITIES if q in res.holdout},
        "weights": weights,
        "unified_holdout": unified["metrics"],
        "market_calibration_holdout": {
            m: {"ece": ece, "table": table}
            for m, (table, ece) in unified["market_calibration"].items()
        },
        "calibration_validation_ece": {
            f"{b}|{fam}|{line}": ece
            for (b, fam, line), (_t, ece) in res.calib_validation.items()
        },
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "brain_competition_A.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"JSON -> {out}")

    # Tabla rápida por stdout (el reporte markdown se arma desde el JSON).
    print("\nCRPS VALIDACION (menor=mejor; * = pierde contra el bobo)")
    header = "cantidad".ljust(12) + "".join(b.rjust(9) for b in BRAINS)
    print(header)
    for q in QUANTITIES:
        row = res.validation.get(q)
        if not row:
            continue
        bobo = row["bobo"]["crps"]
        cells = []
        for b in BRAINS:
            c = row[b]["crps"]
            mark = "*" if (b != "bobo" and c >= bobo) else " "
            cells.append(f"{c:.4f}{mark}".rjust(9))
        print(q.ljust(12) + "".join(cells))


if __name__ == "__main__":
    main()
