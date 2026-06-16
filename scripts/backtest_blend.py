"""Valida el peso del blend Elo+DC para 1X2 (RPS, holdout out-of-sample).

Entrena Elo + Dixon-Coles con datos ANTERIORES a un corte y evalúa el RPS de cada
método (Elo solo, DC solo, blend a varios pesos) en los partidos posteriores. Elige el
peso que minimiza el RPS. Correr cuando entren datos nuevos para revisar W_ELO.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from mundial_bot.collectors.results import load_results  # noqa: E402
from mundial_bot.models.elo_model import EloModel  # noqa: E402
from mundial_bot.models.goals_model import GoalsModel  # noqa: E402

TRAIN_END = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-08-01")
WEIGHTS = [0.0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]


def rps(p: tuple[float, float, float], outcome: int) -> float:
    """Ranked Probability Score 1X2 (orden home, draw, away). outcome ∈ {0,1,2}."""
    a = [0.0, 0.0, 0.0]
    a[outcome] = 1.0
    c_p = c_a = 0.0
    s = 0.0
    for i in range(2):  # r-1 = 2
        c_p += p[i]
        c_a += a[i]
        s += (c_p - c_a) ** 2
    return s / 2.0


def main():
    df = load_results().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] < TRAIN_END]
    test = df[(df["date"] >= TRAIN_END) & (df["date"] < TEST_END)]

    elo = EloModel().fit(train)
    goals = GoalsModel().fit(train[train["date"] >= pd.Timestamp("2016-01-01")])

    scores = {("elo",): [], ("dc",): []}
    for w in WEIGHTS:
        scores[("blend", w)] = []

    n = 0
    for r in test.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if not goals.can_predict(h, a) or h not in elo.ratings or a not in elo.ratings:
            continue
        try:
            m = goals.predict(h, a, neutral=bool(getattr(r, "neutral", False)))
        except Exception:
            continue
        ep = elo.predict(h, a, neutral=bool(getattr(r, "neutral", False)))
        elo_p = (ep.home, ep.draw, ep.away)
        dc_p = (m.home, m.draw, m.away)
        hs, as_ = int(r.home_score), int(r.away_score)
        outcome = 0 if hs > as_ else (1 if hs == as_ else 2)

        scores[("elo",)].append(rps(elo_p, outcome))
        scores[("dc",)].append(rps(dc_p, outcome))
        for w in WEIGHTS:
            bp = tuple(w * e + (1 - w) * d for e, d in zip(elo_p, dc_p, strict=True))
            t = sum(bp)
            scores[("blend", w)].append(rps((bp[0] / t, bp[1] / t, bp[2] / t), outcome))
        n += 1

    print(f"n test = {n}  (train<{TRAIN_END.date()}, test<{TEST_END.date()})\n")
    print(f"Elo solo : RPS {np.mean(scores[('elo',)]):.4f}")
    print(f"DC  solo : RPS {np.mean(scores[('dc',)]):.4f}\n")
    best = None
    for w in WEIGHTS:
        m = float(np.mean(scores[("blend", w)]))
        if best is None or m < best[1]:
            best = (w, m)
        print(f"blend w_elo={w:.2f} : RPS {m:.4f}")
    print(f"\nMEJOR peso: w_elo={best[0]:.2f} (RPS {best[1]:.4f})")


if __name__ == "__main__":
    main()
