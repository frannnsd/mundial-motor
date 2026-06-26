"""Simulador Monte Carlo de un partido sobre los modelos del cerebro.

Toma el xG (Dixon-Coles) + córners/tarjetas/tiros (modelos de conteo) y, opcional,
un ``MatchContext``, y corre el partido N veces para devolver la distribución de
TODO lo que pasa: 1X2, total de goles, marcadores, ambos marcan, valla invicta,
mitades, remontadas, córners/tarjetas/tiros, y narrativas legibles.

Por qué Monte Carlo y no solo la matriz de Dixon-Coles: la matriz da los mercados
de goles exactos (los seguimos usando), pero la simulación da lo que la matriz no
puede sola: la dinámica por mitades (marca en ambos tiempos, remontadas), la
combinación de mercados distintos y el efecto del contexto.

Determinístico: usa un seed fijo (no el reloj), así dos corridas iguales coinciden.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from mundial_bot.models.count_market import _nb_params
from mundial_bot.sim.context import MatchContext

SEED = 20260611  # fijo (arranque del Mundial); reproducible
MATRIX_SIZE = 12
GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
P_FIRST_HALF = 0.46  # algo menos de la mitad de los goles caen en el 1T


def _poisson_matrix(lh: float, la: float, size: int = MATRIX_SIZE) -> np.ndarray:
    """Matriz P[i,j] de marcadores a partir de xG (Poisson independiente)."""
    i = np.arange(size)
    m = np.outer(poisson.pmf(i, max(lh, 1e-6)), poisson.pmf(i, max(la, 1e-6)))
    return m / m.sum()


def _sample_count(rng: np.random.Generator, mean: float, dispersion: float, n: int) -> np.ndarray:
    """Muestrea un conteo (córners/tarjetas/tiros) con Negative Binomial o Poisson."""
    if mean <= 0:
        return np.zeros(n, dtype=int)
    var = mean * max(dispersion, 1.0)
    params = _nb_params(mean, var)
    if params is None:
        return rng.poisson(mean, n)
    r, p = params
    return rng.negative_binomial(max(r, 1e-6), p, n)


def _lines_over(values: np.ndarray, lines) -> dict[str, float]:
    """{línea: P(over)} para una serie de valores simulados."""
    return {f"{ln:g}": round(float((values > ln).mean()), 4) for ln in lines}


def _centered_ladder(mean: float):
    """Escalera de líneas .5 centrada en el valor esperado (remates totales)."""
    base = round(mean)
    return tuple(base + o - 0.5 for o in (-5, -3, -1, 1, 3, 5) if base + o - 0.5 > 0)


def simulate(
    brain,
    home: str,
    away: str,
    *,
    n: int = 10000,
    neutral: bool = True,
    referee: str | None = None,
    knockout: bool = False,
    context: MatchContext | None = None,
) -> dict:
    """Corre la simulación y devuelve un dict JSON-friendly con todas las distribuciones."""
    rh, ra = brain.resolve(home), brain.resolve(away)
    goals = brain.models.goals
    if goals is None or not goals.can_predict(rh, ra):
        raise ValueError(f"No tengo modelo de goles para {home} o {away}.")

    matrix, hx, ax = goals.score_matrix(rh, ra, neutral=neutral)
    ctx = context or MatchContext(knockout=knockout)
    lh, la = hx * ctx.home_attack, ax * ctx.away_attack

    # Si el contexto altera el ataque, reconstruyo la matriz; si no, uso la de Dixon-Coles.
    m = matrix if (ctx.home_attack == 1.0 and ctx.away_attack == 1.0) else _poisson_matrix(lh, la, matrix.shape[0])

    rng = np.random.default_rng(SEED)
    flat = m.flatten()
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, p=flat)
    ncols = m.shape[1]
    hg = idx // ncols
    ag = idx % ncols
    total = hg + ag

    # Mitades (para resultado al descanso, marca en ambos tiempos, remontadas).
    hg1 = rng.binomial(hg, P_FIRST_HALF)
    ag1 = rng.binomial(ag, P_FIRST_HALF)
    hg2, ag2 = hg - hg1, ag - ag1
    ht_home, ht_away = hg1 > ag1, ag1 > hg1
    ft_home, ft_away = hg > ag, ag > hg
    comeback = float(((ht_away & ft_home) | (ht_home & ft_away)).mean())
    both_halves = float((((hg1 + ag1) > 0) & ((hg2 + ag2) > 0)).mean())

    # Marcadores más probables.
    pairs, counts = np.unique(np.stack([hg, ag], axis=1), axis=0, return_counts=True)
    order = np.argsort(-counts)[:6]
    top_scores = [
        {"score": f"{int(pairs[i][0])}-{int(pairs[i][1])}", "prob": round(float(counts[i] / n), 4)}
        for i in order
    ]

    # Distribución del total de goles (0..6+).
    goal_dist = {}
    for g in range(6):
        goal_dist[str(g)] = round(float((total == g).mean()), 4)
    goal_dist["6+"] = round(float((total >= 6).mean()), 4)

    # Conteos de córners / tarjetas / tiros al arco.
    out_markets: dict[str, dict] = {}
    if brain.corners is not None:
        cp = brain.corners.predict(rh, ra)
        corners = _sample_count(rng, cp.total * ctx.tempo_mult, brain.corners.dispersion, n)
        out_markets["corners"] = {
            "expected": round(float(cp.total * ctx.tempo_mult), 2),
            "lines": _lines_over(corners, (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)),
        }
    if brain.cards is not None:
        kp = brain.cards.predict(rh, ra, referee=referee, knockout=knockout)
        cards = _sample_count(rng, kp.total * ctx.cards_mult, brain.cards.dispersion, n)
        out_markets["cards"] = {
            "expected": round(float(kp.total * ctx.cards_mult), 2),
            "referee": referee,
            "lines": _lines_over(cards, (2.5, 3.5, 4.5, 5.5, 6.5)),
        }
    if brain.shots is not None:
        sp = brain.shots.predict(rh, ra)
        shots = _sample_count(rng, sp.total * ctx.tempo_mult, brain.shots.dispersion, n)
        out_markets["shots_on_target"] = {
            "expected": round(float(sp.total * ctx.tempo_mult), 2),
            "lines": _lines_over(shots, (5.5, 6.5, 7.5, 8.5, 9.5, 10.5)),
        }
    if getattr(brain, "total_shots", None) is not None:
        tp = brain.total_shots.predict(rh, ra)
        exp_t = tp.total * ctx.tempo_mult
        rem = _sample_count(rng, exp_t, brain.total_shots.dispersion, n)
        out_markets["shots"] = {
            "expected": round(float(exp_t), 2),
            "lines": _lines_over(rem, _centered_ladder(exp_t)),
        }

    p_home = float(ft_home.mean())
    p_away = float(ft_away.mean())
    p_draw = float((hg == ag).mean())

    fav = rh if p_home >= max(p_draw, p_away) else (ra if p_away >= p_draw else "Empate")
    narratives = _narratives(rh, ra, p_home, p_draw, p_away, top_scores, out_markets, comeback, fav)
    explain = _build_explain(rh, ra, float(hx), float(ax), hg, ag, out_markets, referee, ctx, brain)

    return {
        "home": rh,
        "away": ra,
        "n": n,
        "result": {
            "home": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away": round(p_away, 4),
        },
        "exp_goals": {"home": round(float(hg.mean()), 2), "away": round(float(ag.mean()), 2)},
        "model_xg": {"home": round(float(hx), 2), "away": round(float(ax), 2)},
        "goal_dist": goal_dist,
        "over_lines": _lines_over(total, GOAL_LINES),
        "btts": round(float(((hg > 0) & (ag > 0)).mean()), 4),
        "clean_sheet": {
            "home": round(float((ag == 0).mean()), 4),
            "away": round(float((hg == 0).mean()), 4),
        },
        "halves": {
            "first_half_goals": round(float((hg1 + ag1).mean()), 2),
            "goal_both_halves": round(both_halves, 4),
            "comeback": round(comeback, 4),
        },
        "top_scores": top_scores,
        "markets": out_markets,
        "favorite": fav,
        "narratives": narratives,
        "explain": explain,
        "context_notes": ctx.notes,
        "context_applied": not ctx.neutral,
    }


def _build_explain(home, away, hx, ax, hg, ag, markets, referee, ctx, brain) -> list[dict]:
    """Explica el PORQUÉ de cada número de la simulación, con los datos del modelo."""
    tot = hx + ax
    p_home_scores = float((hg > 0).mean())
    p_away_scores = float((ag > 0).mean())
    items: list[dict] = [
        {
            "title": "Resultado (1X2)",
            "text": (
                f"Sale de simular 8.000 partidos con el xG del modelo: {home} {hx:.2f} vs "
                f"{away} {ax:.2f} (Dixon-Coles sobre 50k partidos internacionales + la forma "
                f"del Mundial). El que genera más xG gana más seguido."
            ),
        },
        {
            "title": "Goles",
            "text": (
                f"Se esperan ~{tot:.1f} goles (la suma de los xG). Con esa media, lo normal son "
                f"2-3 goles; de ahí salen el Más/Menos y los marcadores más probables."
            ),
        },
        {
            "title": "Ambos marcan",
            "text": (
                f"{home} mete al menos un gol en {p_home_scores:.0%} de las simulaciones y "
                f"{away} en {p_away_scores:.0%}; que pasen las dos cosas juntas es el BTTS."
            ),
        },
    ]
    if "corners" in markets:
        cal = getattr(brain.corners, "calibration", 1.0)
        items.append({
            "title": "Córners",
            "text": (
                f"~{markets['corners']['expected']:.1f} córners totales. Modelo ataque/defensa: "
                f"cuántos genera cada equipo por cuántos concede el rival, auto-calibrado contra "
                f"los over reales (factor {cal:.2f})."
            ),
        })
    if "cards" in markets:
        ref_txt = ""
        refmap = getattr(brain.cards, "referee_cards", None)
        if referee and refmap and referee in refmap:
            ref_txt = f" El árbitro {referee} promedia {refmap[referee]:.1f} tarjetas por partido."
        kn = " Es eliminación directa, lo que suma fricción." if ctx.knockout else ""
        items.append({
            "title": "Tarjetas",
            "text": (
                f"~{markets['cards']['expected']:.1f} tarjetas. El driver más fuerte es el "
                f"árbitro, después la disciplina de los equipos.{ref_txt}{kn}"
            ),
        })
    if "shots_on_target" in markets:
        items.append({
            "title": "Tiros al arco",
            "text": (
                f"~{markets['shots_on_target']['expected']:.1f} tiros al arco totales, del "
                f"modelo ataque/defensa de tiros al arco (auto-calibrado)."
            ),
        })
    if "shots" in markets:
        items.append({
            "title": "Remates",
            "text": (
                f"~{markets['shots']['expected']:.1f} remates totales (al arco + afuera). "
                f"Modelo de remates ataque/defensa: cuántos patea cada equipo por cuántos "
                f"permite el rival."
            ),
        })
    if ctx.notes:
        items.append({"title": "Contexto aplicado", "text": " ".join(ctx.notes)})
    return items


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _narratives(rh, ra, p_home, p_draw, p_away, top_scores, markets, comeback, fav) -> list[str]:
    """Arma 3-4 frases legibles con lo más jugoso de la simulación."""
    out: list[str] = []
    if fav == "Empate":
        out.append(f"Partido parejísimo: el empate es lo más probable ({_pct(p_draw)}).")
    else:
        p = p_home if fav == rh else p_away
        out.append(f"Lo más probable: gana {fav} ({_pct(p)}).")
    if top_scores:
        ts = top_scores[0]
        out.append(f"Marcador top: {ts['score']} ({_pct(ts['prob'])}).")
    if "corners" in markets:
        out.append(f"Córners esperados: ~{markets['corners']['expected']:.1f} en total.")
    if comeback >= 0.12:
        out.append(f"Ojo a la remontada: en {_pct(comeback)} de las simulaciones, el que va perdiendo al descanso lo da vuelta.")
    return out
