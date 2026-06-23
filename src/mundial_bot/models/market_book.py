"""Libro de mercados COMPLETO de un partido: el "cerebro matemático" del bot.

De la matriz de marcadores de Dixon-Coles (P[i,j] = P(local i, visita j)) se deriva la
probabilidad de TODOS los mercados que ofrecen las casas — no solo ganador/goles:
1X2, doble oportunidad, empate-no-apuesta, hándicap asiático (toda la escalera),
totales (medias y enteras con push), totales por equipo, ambos marcan, par/impar,
valla invicta, gana a cero, marcador exacto y goles exactos. Más córners y tarjetas
desde sus distribuciones (Negative Binomial).

Cada selección lleva su `odds_key` = (mercado, resultado) en el formato de API-Football,
para poder pegarle la CUOTA REAL de la casa. Así Claude —el otro cerebro— ve, de cada
mercado, la probabilidad del modelo y lo que paga la casa, y razona el panorama entero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mundial_bot.collectors.odds_af import MarketOdds
from mundial_bot.models.blend import blend_1x2
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.count_market import over_under
from mundial_bot.models.elo_model import EloModel
from mundial_bot.models.goals_model import GoalsModel, GoalsModelError

# Escaleras de líneas que ofrecen las casas.
_AH_LINES = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
_TOTAL_HALF = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
_TOTAL_WHOLE = (1.0, 2.0, 3.0, 4.0)
_TEAM_TOTAL = (0.5, 1.5, 2.5)
_CORNER_LADDER = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
_CARD_LADDER = (2.5, 3.5, 4.5, 5.5, 6.5)
_SHOT_LADDER = (5.5, 6.5, 7.5, 8.5, 9.5, 10.5)


@dataclass(frozen=True)
class Selection:
    """Una apuesta concreta con su probabilidad y cuota justa (considera el push)."""

    market: str        # grupo, ej. "Hándicap asiático"
    pick: str          # ej. "Brazil -1.5"
    prob: float        # prob. efectiva de ganar (condicional a que no haya push/devolución)
    fair: float        # cuota decimal del modelo = (1 - push) / prob_ganar
    push: float = 0.0  # prob. de empate-devolución (líneas enteras)
    note: str = ""     # el porqué, corto
    odds_key: tuple[str, str] | None = None  # (mercado, resultado) en API-Football


@dataclass(frozen=True)
class MarketBook:
    """Panorama completo del partido: drivers del modelo + todas las selecciones."""

    match: str
    home: str
    away: str
    home_xg: float
    away_xg: float
    p_home: float          # 1X2 CANÓNICO = blend Elo+DC (lo que usan las selecciones)
    p_draw: float
    p_away: float
    exp_goals: float
    elo_home: float = 0.0  # 1X2 según Elo (para mostrar el desglose)
    elo_draw: float = 0.0
    elo_away: float = 0.0
    dc_home: float = 0.0   # 1X2 implícito por la matriz Dixon-Coles (de goles)
    dc_draw: float = 0.0
    dc_away: float = 0.0
    selections: list[Selection] = field(default_factory=list)

    def by_market(self) -> dict[str, list[Selection]]:
        out: dict[str, list[Selection]] = {}
        for s in self.selections:
            out.setdefault(s.market, []).append(s)
        return out


def _sel(
    market: str, pick: str, p_win: float, *,
    push: float = 0.0, note: str = "", odds_key: tuple[str, str] | None = None,
) -> Selection:
    """Arma una Selection con cuota del modelo push-aware. p_win = prob bruta de ganar."""
    live = 1.0 - push
    eff = p_win / live if live > 1e-9 else 0.0
    fair = round(live / p_win, 2) if p_win > 1e-9 else 0.0
    return Selection(market=market, pick=pick, prob=eff, fair=fair, push=push,
                     note=note, odds_key=odds_key)


def _ah_sign(h: float) -> str:
    """Signo de la línea como lo escribe API-Football: '+0', '-0.5', '-1', '+1.5'."""
    if h > 0:
        return f"+{h:g}"
    if h == 0:
        return "+0"
    return f"{h:g}"


def _goals_selections(
    matrix: np.ndarray, home: str, away: str, home_xg: float, away_xg: float,
    win_probs: tuple[float, float, float] | None = None,
) -> list[Selection]:
    """Deriva todos los mercados de goles desde la matriz de marcadores.

    `win_probs` (home, draw, away): si se pasa (ej. el blend Elo+DC), se usa para los
    mercados de ganar/empatar/perder (1X2, doble oportunidad, empate-no-apuesta). El
    resto (hándicaps, totales, etc.) siempre sale de la matriz de goles.
    """
    n = matrix.shape[0]
    idx = np.arange(n)
    i = idx.reshape(-1, 1)
    j = idx.reshape(1, -1)
    margin = i - j           # local − visita
    total = i + j
    exp_goals = home_xg + away_xg
    fav = home if home_xg >= away_xg else away
    out: list[Selection] = []

    # --- 1X2 (usa el blend si se pasó; si no, la matriz DC) ---
    p_home, p_draw, p_away = win_probs or (
        float(matrix[margin > 0].sum()),
        float(matrix[margin == 0].sum()),
        float(matrix[margin < 0].sum()),
    )
    drv = f"xG {home_xg:.1f}-{away_xg:.1f}"
    out += [
        _sel("Ganador (1X2)", f"Gana {home}", p_home, note=drv,
             odds_key=("Match Winner", "Home")),
        _sel("Ganador (1X2)", "Empate", p_draw, note=drv,
             odds_key=("Match Winner", "Draw")),
        _sel("Ganador (1X2)", f"Gana {away}", p_away, note=drv,
             odds_key=("Match Winner", "Away")),
    ]

    # --- Doble oportunidad ---
    out += [
        _sel("Doble oportunidad", f"{home} o empate", p_home + p_draw,
             odds_key=("Double Chance", "Home/Draw")),
        _sel("Doble oportunidad", f"{home} o {away}", p_home + p_away,
             odds_key=("Double Chance", "Home/Away")),
        _sel("Doble oportunidad", f"empate o {away}", p_draw + p_away,
             odds_key=("Double Chance", "Draw/Away")),
    ]

    # --- Empate no apuesta (push en empate) ---
    out += [
        _sel("Empate no apuesta", f"{home}", p_home, push=p_draw,
             odds_key=("Draw No Bet", "Home")),
        _sel("Empate no apuesta", f"{away}", p_away, push=p_draw,
             odds_key=("Draw No Bet", "Away")),
    ]

    # --- Hándicap asiático (local; el visitante toma la línea opuesta) ---
    for h in _AH_LINES:
        win = float(matrix[(margin + h) > 0].sum())
        push = float(matrix[(margin + h) == 0].sum()) if float(h).is_integer() else 0.0
        out.append(_sel("Hándicap asiático", f"{home} {_ah_sign(h)}", win, push=push,
                        note=f"fav {fav}", odds_key=("Asian Handicap", f"Home {_ah_sign(h)}")))
    for h in _AH_LINES:
        win = float(matrix[(-margin + h) > 0].sum())
        push = float(matrix[(-margin + h) == 0].sum()) if float(h).is_integer() else 0.0
        out.append(_sel("Hándicap asiático", f"{away} {_ah_sign(h)}", win, push=push,
                        note=f"fav {fav}", odds_key=("Asian Handicap", f"Away {_ah_sign(h)}")))

    # --- Totales (medias: sin push) ---
    tnote = f"~{exp_goals:.1f} goles esperados"
    for line in _TOTAL_HALF:
        over = float(matrix[total > line].sum())
        out.append(_sel("Goles Más/Menos", f"Más de {line:g}", over, note=tnote,
                        odds_key=("Goals Over/Under", f"Over {line:g}")))
        out.append(_sel("Goles Más/Menos", f"Menos de {line:g}", 1.0 - over, note=tnote,
                        odds_key=("Goals Over/Under", f"Under {line:g}")))
    # --- Totales enteras (con push) ---
    for line in _TOTAL_WHOLE:
        over = float(matrix[total > line].sum())
        push = float(matrix[total == line].sum())
        under = float(matrix[total < line].sum())
        out.append(_sel("Goles asiáticos", f"Más de {line:g}", over, push=push, note=tnote,
                        odds_key=("Goals Over/Under", f"Over {line:g}")))
        out.append(_sel("Goles asiáticos", f"Menos de {line:g}", under, push=push, note=tnote,
                        odds_key=("Goals Over/Under", f"Under {line:g}")))

    # --- Totales por equipo ---
    home_dist = matrix.sum(axis=1)   # goles del local
    away_dist = matrix.sum(axis=0)   # goles de la visita
    teams = ((home, home_dist, home_xg, "Total - Home"), (away, away_dist, away_xg, "Total - Away"))
    for team, dist, xg, mkt in teams:
        nt = f"~{xg:.1f} del equipo"
        for line in _TEAM_TOTAL:
            over = float(dist[idx > line].sum())
            out.append(_sel("Total por equipo", f"{team} Más de {line:g}", over, note=nt,
                            odds_key=(mkt, f"Over {line:g}")))
            out.append(_sel("Total por equipo", f"{team} Menos de {line:g}", 1.0 - over, note=nt,
                            odds_key=(mkt, f"Under {line:g}")))

    # --- Ambos marcan ---
    btts_yes = float(matrix[(i >= 1) & (j >= 1)].sum())
    out += [
        _sel("Ambos marcan", "Sí", btts_yes, note=tnote, odds_key=("Both Teams Score", "Yes")),
        _sel("Ambos marcan", "No", 1.0 - btts_yes, note=tnote,
             odds_key=("Both Teams Score", "No")),
    ]

    # --- Par / Impar (total de goles) ---
    even = float(matrix[(total % 2) == 0].sum())
    out += [
        _sel("Par/Impar", "Par", even, odds_key=("Odd/Even", "Even")),
        _sel("Par/Impar", "Impar", 1.0 - even, odds_key=("Odd/Even", "Odd")),
    ]

    # --- Valla invicta / gana a cero ---
    cs_home = float(matrix[:, 0].sum())   # visita no marca
    cs_away = float(matrix[0, :].sum())   # local no marca
    wtn_home = float(matrix[(margin > 0) & (j == 0)].sum())
    wtn_away = float(matrix[(margin < 0) & (i == 0)].sum())
    out += [
        _sel("Valla invicta", f"{home} sin recibir", cs_home,
             odds_key=("Clean Sheet - Home", "Yes")),
        _sel("Valla invicta", f"{away} sin recibir", cs_away,
             odds_key=("Clean Sheet - Away", "Yes")),
        _sel("Gana a cero", f"{home} gana sin recibir", wtn_home,
             odds_key=("Win to Nil - Home", "Yes")),
        _sel("Gana a cero", f"{away} gana sin recibir", wtn_away,
             odds_key=("Win to Nil - Away", "Yes")),
    ]

    # --- Goles exactos (0,1,2,3,4,5+) — sin cuota mapeada limpia ---
    for n_goals in range(5):
        out.append(_sel("Goles exactos", f"{n_goals}", float(matrix[total == n_goals].sum())))
    out.append(_sel("Goles exactos", "5+", float(matrix[total >= 5].sum())))

    # --- Marcador exacto (top 6) ---
    flat = [(matrix[a, b], a, b) for a in range(n) for b in range(n)]
    flat.sort(reverse=True)
    for prob, a, b in flat[:6]:
        out.append(_sel("Marcador exacto", f"{a}-{b}", float(prob),
                        odds_key=("Exact Score", f"{a}:{b}")))

    return out


def _count_selections(
    label: str, total: float, variance: float, ladder: tuple[float, ...], odds_market: str
) -> list[Selection]:
    """Over/Under de un mercado de conteo (córners/tarjetas) en toda su escalera."""
    note = f"~{total:.1f} esperados"
    out: list[Selection] = []
    for line in ladder:
        over, under = over_under(total, line, variance=variance)
        out.append(_sel(label, f"Más de {line:g}", over, note=note,
                        odds_key=(odds_market, f"Over {line:g}")))
        out.append(_sel(label, f"Menos de {line:g}", under, note=note,
                        odds_key=(odds_market, f"Under {line:g}")))
    return out


def build_market_book(
    home: str,
    away: str,
    *,
    elo: EloModel,
    goals: GoalsModel | None,
    corners: CornersModel | None = None,
    cards=None,
    shots=None,
    referee: str | None = None,
    knockout: bool = False,
    neutral: bool = True,
    match_name: str | None = None,
) -> MarketBook:
    """Arma el libro de mercados completo de un partido (todos los mercados + cuota del modelo)."""
    p = elo.predict(home, away, neutral=neutral)
    elo_p = (p.home, p.draw, p.away)
    selections: list[Selection] = []
    home_xg = away_xg = 0.0
    dc_home, dc_draw, dc_away = elo_p
    blended = elo_p

    if goals is not None and goals.can_predict(home, away):
        try:
            matrix, home_xg, away_xg = goals.score_matrix(home, away, neutral=neutral)
            n = matrix.shape[0]
            m = np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1)
            dc_home = float(matrix[m > 0].sum())
            dc_draw = float(matrix[m == 0].sum())
            dc_away = float(matrix[m < 0].sum())
            blended = blend_1x2(elo_p, (dc_home, dc_draw, dc_away))
            selections += _goals_selections(
                matrix, home, away, home_xg, away_xg, win_probs=blended
            )
        except GoalsModelError:
            pass

    if corners is not None:
        cp = corners.predict(home, away)
        selections += _count_selections(
            "Córners Más/Menos", cp.total, cp.total * corners.dispersion,
            _CORNER_LADDER, "Corners Over Under",
        )

    if cards is not None:
        cdp = cards.predict(home, away, referee=referee, knockout=knockout)
        variance = cdp.total * getattr(cards, "dispersion", 1.0)
        selections += _count_selections(
            "Tarjetas Más/Menos", cdp.total, variance, _CARD_LADDER, "Cards Over/Under",
        )

    if shots is not None:
        sp = shots.predict(home, away)
        selections += _count_selections(
            "Tiros al arco Más/Menos", sp.total, sp.total * shots.dispersion,
            _SHOT_LADDER, "Total Shots on Target",
        )

    return MarketBook(
        match=match_name or f"{home} vs {away}",
        home=home, away=away, home_xg=home_xg, away_xg=away_xg,
        p_home=blended[0], p_draw=blended[1], p_away=blended[2],
        exp_goals=home_xg + away_xg,
        elo_home=elo_p[0], elo_draw=elo_p[1], elo_away=elo_p[2],
        dc_home=dc_home, dc_draw=dc_draw, dc_away=dc_away,
        selections=selections,
    )


def real_odd(sel: Selection, odds: dict[str, MarketOdds] | None) -> tuple[float, str] | None:
    """Cuota REAL de la casa para una selección (vía su odds_key). None si no está."""
    if not odds or not sel.odds_key:
        return None
    mo = odds.get(sel.odds_key[0])
    if mo and sel.odds_key[1] in mo.best:
        return mo.best[sel.odds_key[1]]
    return None


def format_market_book(
    book: MarketBook, *, odds: dict[str, MarketOdds] | None = None, min_prob: float = 0.0
) -> str:
    """Texto plano del libro completo. Si se pasan `odds`, muestra la cuota REAL de la casa."""
    head = (
        f"PANORAMA — {book.match}\n"
        f"1X2 FINAL (blend Elo+DC, el que vale): {book.home} {book.p_home:.0%} / "
        f"X {book.p_draw:.0%} / {book.away} {book.p_away:.0%}\n"
        f"   desglose → Elo: {book.elo_home:.0%}/{book.elo_draw:.0%}/{book.elo_away:.0%} · "
        f"DC: {book.dc_home:.0%}/{book.dc_draw:.0%}/{book.dc_away:.0%}\n"
        f"xG modelo: {book.home} {book.home_xg:.2f} − {book.away} {book.away_xg:.2f} "
        f"(total ~{book.exp_goals:.2f})\n"
        "Nota: el 1X2/doble oportunidad/empate-no-apuesta usan el BLEND; los goles, "
        "hándicaps y totales salen de la matriz Dixon-Coles. 'modelo X%' = prob del "
        "modelo; 'casa Y.YY' = lo que paga.\n"
    )
    lines = [head]
    for market, sels in book.by_market().items():
        shown = [s for s in sels if s.prob >= min_prob]
        if not shown:
            continue
        lines.append(f"\n[{market}]")
        for s in shown:
            extra = f" · push {s.push:.0%}" if s.push > 0.01 else ""
            found = real_odd(s, odds)
            casa = f" · casa {found[0]:.2f} ({found[1]})" if found else ""
            lines.append(
                f"  {s.pick}: modelo {s.prob:.0%} (cuota s/modelo {s.fair:.2f}){casa}{extra}"
            )
    return "\n".join(lines)
