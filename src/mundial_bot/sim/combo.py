"""Combinada del MISMO partido para un VALOR (cuota) objetivo.

Franco pide una cuota (ej. @5.00) y armamos la combinada de ese partido cuya cuota
combinada se ACERCA a ese valor, eligiendo las patas MÁS probables posibles. Las patas
de goles usan probabilidad CONJUNTA exacta (correlación) desde la matriz de marcadores;
córners/tarjetas/tiros se multiplican como independientes.

Busca por fuerza bruta (todas las combinaciones de 1 pata por familia) la que mejor
matchea la cuota pedida, maximizando la probabilidad conjunta (la combinada más probable
para ese pago).
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from mundial_bot.models.count_market import CARD_LINES, CORNER_LINES, best_line, over_under
from mundial_bot.models.shots_model import SHOT_LINES


def _goals_options(rh: str, ra: str, matrix: np.ndarray):
    """Opciones de patas de goles (familia, máscara booleana, descripción, odds_key)."""
    n = matrix.shape[0]
    idx = np.arange(n)
    i = idx.reshape(-1, 1)
    j = idx.reshape(1, -1)
    margin = i - j
    total = i + j
    home_fav = float(matrix[margin > 0].sum()) >= float(matrix[margin < 0].sum())

    opts = []
    if home_fav:
        opts += [
            ("margin", margin >= 0, f"{rh} o empate", ("Double Chance", "Home/Draw")),
            ("margin", margin > 0, f"Gana {rh}", ("Match Winner", "Home")),
            ("margin", margin > 1, f"{rh} -1.5", ("Asian Handicap", "Home -1.5")),
            ("margin", margin > 2, f"{rh} -2.5", ("Asian Handicap", "Home -2.5")),
        ]
    else:
        opts += [
            ("margin", margin <= 0, f"empate o {ra}", ("Double Chance", "Draw/Away")),
            ("margin", margin < 0, f"Gana {ra}", ("Match Winner", "Away")),
            ("margin", margin < -1, f"{ra} -1.5", ("Asian Handicap", "Away -1.5")),
            ("margin", margin < -2, f"{ra} -2.5", ("Asian Handicap", "Away -2.5")),
        ]
    for line in (1.5, 2.5, 3.5):
        opts.append(("total", total > line, f"Más de {line:g} goles", ("Goals Over/Under", f"Over {line:g}")))
    for line in (2.5, 3.5):
        opts.append(("total", total < line, f"Menos de {line:g} goles", ("Goals Over/Under", f"Under {line:g}")))
    yes = (i >= 1) & (j >= 1)
    opts.append(("btts", yes, "Ambos marcan: Sí", ("Both Teams Score", "Yes")))
    opts.append(("btts", ~yes, "Ambos marcan: No", ("Both Teams Score", "No")))
    return opts


def _count_options(brain, rh: str, ra: str):
    """Opciones de córners/tarjetas/tiros (familia, prob, descripción, odds_key)."""
    opts = []
    specs = [
        (getattr(brain, "corners", None), CORNER_LINES, "Córners", "Corners Over Under"),
        (getattr(brain, "cards", None), CARD_LINES, "Tarjetas", "Cards Over/Under"),
        (getattr(brain, "shots", None), SHOT_LINES, "Tiros al arco", "Total ShotOnGoal"),
    ]
    for model, ladder, label, mkt in specs:
        if model is None:
            continue
        pred = model.predict(rh, ra)
        var = pred.total * getattr(model, "dispersion", 1.0)
        line = best_line(pred.total, ladder, variance=var)
        po, pu = over_under(pred.total, line, variance=var)
        fam = label
        opts.append((fam, po, f"{label} Más de {line:g}", (mkt, f"Over {line:g}")))
        opts.append((fam, pu, f"{label} Menos de {line:g}", (mkt, f"Under {line:g}")))
    return opts


def _odd(odds, key):
    if not odds or not key:
        return None
    mo = odds.get(key[0])
    if mo and key[1] in mo.best:
        return mo.best[key[1]]
    return None


def build_combo(brain, rh: str, ra: str, target: float, *, odds=None, max_legs: int = 6) -> dict:
    """Arma la combinada del partido cuya cuota se acerca a `target`, lo más probable posible."""
    goals = brain.models.goals
    matrix, _, _ = goals.score_matrix(rh, ra, neutral=True)
    mass = float(matrix.sum())

    gopts = _goals_options(rh, ra, matrix)
    copts = _count_options(brain, rh, ra)

    fam_goals: dict[str, list] = {}
    for fam, mask, desc, key in gopts:
        fam_goals.setdefault(fam, []).append((mask, desc, key))
    fam_count: dict[str, list] = {}
    for fam, prob, desc, key in copts:
        fam_count.setdefault(fam, []).append((prob, desc, key))

    goals_choices = [[None] + v for v in fam_goals.values()]
    count_choices = [[None] + v for v in fam_count.values()]

    best = None  # (score, combined, fair, glegs, clegs)
    for gsel in product(*goals_choices):
        gmask = None
        glegs = []
        for choice in gsel:
            if choice is None:
                continue
            mask, desc, key = choice
            gmask = mask if gmask is None else (gmask & mask)
            glegs.append((desc, key, float(matrix[mask].sum()) / mass))
        goals_joint = float(matrix[gmask].sum()) / mass if gmask is not None else 1.0
        if goals_joint <= 1e-7 and glegs:
            continue
        for csel in product(*count_choices):
            cprob = 1.0
            clegs = []
            for choice in csel:
                if choice is None:
                    continue
                prob, desc, key = choice
                cprob *= prob
                clegs.append((desc, key, prob))
            n_legs = len(glegs) + len(clegs)
            if n_legs < 2 or n_legs > max_legs:
                continue
            combined = goals_joint * cprob
            if combined <= 1e-6:
                continue
            fair = 1.0 / combined
            score = abs(math.log(fair / target))
            if best is None or score < best[0] - 1e-9 or (
                abs(score - best[0]) < 1e-9 and combined > best[1]
            ):
                best = (score, combined, fair, glegs, clegs)

    if best is None:
        return {"target": target, "legs": [], "combined_prob": 0.0, "fair_odds": 0.0,
                "bet365_odds": None, "n_legs": 0}

    _, combined, fair, glegs, clegs = best
    legs = []
    bet = 1.0
    have_all = True
    for desc, key, prob in [*glegs, *clegs]:
        ro = _odd(odds, key)
        if ro:
            bet *= ro[0]
        else:
            have_all = False
        legs.append({
            "desc": desc, "prob": round(prob, 4),
            "odd": round(ro[0], 2) if ro else None, "book": ro[1] if ro else None,
        })
    return {
        "target": target,
        "legs": legs,
        "combined_prob": round(combined, 4),
        "fair_odds": round(fair, 2),
        "bet365_odds": round(bet, 2) if (have_all and legs) else None,
        "n_legs": len(legs),
    }
