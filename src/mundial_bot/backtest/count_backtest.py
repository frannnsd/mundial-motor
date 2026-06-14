"""Backtest walk-forward de córners y tarjetas (validar si tienen edge).

Para cada partido (ordenados por fecha) entrena el modelo SOLO con partidos
anteriores, predice over/under, y compara con el total real. Reporta accuracy,
Brier y log-loss, y los compara contra un baseline ingenuo (predecir según la
media histórica vs la línea). Si el modelo no le gana al ingenuo, no tiene edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.count_market import over_under


@dataclass
class CountBacktestResult:
    market: str
    n: int
    accuracy: float
    naive_accuracy: float
    brier: float
    log_loss: float

    @property
    def edge_vs_naive(self) -> float:
        return self.accuracy - self.naive_accuracy

    def summary(self) -> str:
        verdict = "✅ tiene edge" if self.edge_vs_naive > 0.01 else "⚠️ no le gana al baseline"
        return (
            f"{self.market}: n={self.n} · acc={self.accuracy:.1%} "
            f"(ingenuo {self.naive_accuracy:.1%}, {self.edge_vs_naive:+.1%}) · "
            f"Brier={self.brier:.3f} · logloss={self.log_loss:.3f} → {verdict}"
        )


def _per_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por partido: home/away/referee/total córners/total tarjetas/fecha."""
    rows = []
    for mid, g in df.groupby("match_id"):
        home_rows = g[g.get("is_home", 0) == 1]
        hr = home_rows.iloc[0] if len(home_rows) else g.iloc[0]
        rows.append({
            "match_id": mid,
            "date": hr["date"],
            "home": hr["team"],
            "away": hr["opponent"],
            "referee": hr.get("referee"),
            "corners_total": float(hr["corners_for"] + hr["corners_against"]),
            "cards_total": float(g["cards"].sum()),
        })
    return pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _score(p_over: float, over: bool) -> tuple[int, float, float]:
    pick_correct = int((p_over >= 0.5) == over)
    actual = 1.0 if over else 0.0
    brier = (p_over - actual) ** 2
    p = min(max(p_over if over else 1 - p_over, 1e-12), 1.0)
    return pick_correct, brier, -math.log(p)


def _backtest(
    df: pd.DataFrame, market: str, *, start_frac: float, min_train: int,
    fixed_line: float | None = None,
) -> CountBacktestResult:
    matches = _per_match_table(df)
    total_col = "corners_total" if market == "córners" else "cards_total"
    start = int(len(matches) * start_frac)

    n = correct = naive_correct = 0
    brier_sum = ll_sum = 0.0
    for i in range(start, len(matches)):
        m = matches.iloc[i]
        train = df[df["date"] < m["date"]]
        if len(train) < min_train:
            continue
        train_matches = matches[matches["date"] < m["date"]]
        if len(train_matches) < 20:
            continue

        if market == "córners":
            model = CornersModel.from_events(train)
            pred = model.predict(m["home"], m["away"])
        else:
            model = CardsModel.from_events(train)
            pred = model.predict(m["home"], m["away"], referee=m["referee"])

        # A línea fija evaluamos la estimación de total del modelo (no la línea más
        # cercana, que siempre queda ~50/50).
        if fixed_line is not None:
            line = fixed_line
            p_over, _ = over_under(pred.total, line, variance=pred.total * model.dispersion)
        else:
            line, p_over = pred.line, pred.p_over

        actual_total = m[total_col]
        over = actual_total > line
        c, brier, ll = _score(p_over, over)
        correct += c
        brier_sum += brier
        ll_sum += ll

        # Baseline ingenuo: predecir según la media histórica vs la misma línea.
        naive_over = train_matches[total_col].mean() > line
        naive_correct += int(naive_over == over)
        n += 1

    if n == 0:
        raise ValueError("Sin partidos suficientes para backtestear.")
    return CountBacktestResult(
        market=market, n=n,
        accuracy=correct / n, naive_accuracy=naive_correct / n,
        brier=brier_sum / n, log_loss=ll_sum / n,
    )


def backtest_corners(
    df: pd.DataFrame, *, start_frac: float = 0.4, min_train: int = 100,
    fixed_line: float | None = None,
) -> CountBacktestResult:
    return _backtest(df, "córners", start_frac=start_frac, min_train=min_train,
                     fixed_line=fixed_line)


def backtest_cards(
    df: pd.DataFrame, *, start_frac: float = 0.4, min_train: int = 100,
    fixed_line: float | None = None,
) -> CountBacktestResult:
    return _backtest(df, "tarjetas", start_frac=start_frac, min_train=min_train,
                     fixed_line=fixed_line)
