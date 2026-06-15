"""Libro de mercados COMPLETO de un partido: el "cerebro matemático" del bot.

De la matriz de marcadores de Dixon-Coles (P[i,j] = P(local i, visita j)) se deriva la
probabilidad de TODOS los mercados que ofrecen las casas — no solo ganador/goles:
1X2, doble oportunidad, empate-no-apuesta, hándicap asiático (toda la escalera),
totales (medias y enteras con push), totales por equipo, ambos marcan, par/impar,
valla invicta, gana a cero, marcador exacto y goles exactos. Más córners y tarjetas
desde sus distribuciones (Negative Binomial).

Cada selección trae su probabilidad efectiva y su CUOTA JUSTA (push-aware): así Claude
—el otro cerebro— razona sobre el panorama entero y explica el porqué de cada chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mundial_bot.models.cards_model import CardsModel
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


@dataclass(frozen=True)
class Selection:
    """Una apuesta concreta con su probabilidad y cuota justa (considera el push)."""

    market: str        # grupo, ej. "Hándicap asiático"
    pick: str          # ej. "Brazil -1.5"
    prob: float        # prob. efectiva de ganar (condicional a que no haya push/devolución)
    fair: float        # cuota decimal justa = (1 - push) / prob_ganar
    push: float = 0.0  # prob. de empate-devolución (líneas enteras)
    note: str = ""     # el porqué, corto


@dataclass(frozen=True)
class MarketBook:
    """Panorama completo del partido: drivers del modelo + todas las selecciones."""

    match: str
    home: str
    away: str
    home_xg: float
    away_xg: float
    p_home: float          # 1X2 según Elo (mejor baseline de ganador en data rala)
    p_draw: float
    p_away: float
    exp_goals: float
    dc_home: float = 0.0   # 1X2 implícito por la matriz Dixon-Coles (de goles)
    dc_draw: float = 0.0
    dc_away: float = 0.0
    selections: list[Selection] = field(default_factory=list)

    def by_market(self) -> dict[str, list[Selection]]:
        out: dict[str, list[Selection]] = {}
        for s in self.selections:
            out.setdefault(s.market, []).append(s)
        return out


def _sel(market: str, pick: str, p_win: float, *, push: float = 0.0, note: str = "") -> Selection:
    """Arma una Selection con cuota justa push-aware. p_win = prob bruta de ganar."""
    live = 1.0 - push
    eff = p_win / live if live > 1e-9 else 0.0
    fair = round(live / p_win, 2) if p_win > 1e-9 else 0.0
    return Selection(market=market, pick=pick, prob=eff, fair=fair, push=push, note=note)


def _goals_selections(
    matrix: np.ndarray, home: str, away: str, home_xg: float, away_xg: float
) -> list[Selection]:
    """Deriva todos los mercados de goles desde la matriz de marcadores."""
    n = matrix.shape[0]
    idx = np.arange(n)
    i = idx.reshape(-1, 1)
    j = idx.reshape(1, -1)
    margin = i - j           # local − visita
    total = i + j
    exp_goals = home_xg + away_xg
    fav = home if home_xg >= away_xg else away
    out: list[Selection] = []

    # --- 1X2 ---
    p_home = float(matrix[margin > 0].sum())
    p_draw = float(matrix[margin == 0].sum())
    p_away = float(matrix[margin < 0].sum())
    drv = f"xG {home_xg:.1f}-{away_xg:.1f}"
    out += [
        _sel("Ganador (1X2)", f"Gana {home}", p_home, note=drv),
        _sel("Ganador (1X2)", "Empate", p_draw, note=drv),
        _sel("Ganador (1X2)", f"Gana {away}", p_away, note=drv),
    ]

    # --- Doble oportunidad ---
    out += [
        _sel("Doble oportunidad", f"{home} o empate", p_home + p_draw),
        _sel("Doble oportunidad", f"{home} o {away}", p_home + p_away),
        _sel("Doble oportunidad", f"empate o {away}", p_draw + p_away),
    ]

    # --- Empate no apuesta (push en empate) ---
    out += [
        _sel("Empate no apuesta", f"{home}", p_home, push=p_draw),
        _sel("Empate no apuesta", f"{away}", p_away, push=p_draw),
    ]

    # --- Hándicap asiático (local; el visitante toma la línea opuesta) ---
    for h in _AH_LINES:
        win = float(matrix[(margin + h) > 0].sum())
        push = float(matrix[(margin + h) == 0].sum()) if float(h).is_integer() else 0.0
        sign = f"+{h:g}" if h > 0 else f"{h:g}"
        out.append(_sel("Hándicap asiático", f"{home} {sign}", win, push=push, note=f"fav {fav}"))
    for h in _AH_LINES:
        win = float(matrix[(-margin + h) > 0].sum())
        push = float(matrix[(-margin + h) == 0].sum()) if float(h).is_integer() else 0.0
        sign = f"+{h:g}" if h > 0 else f"{h:g}"
        out.append(_sel("Hándicap asiático", f"{away} {sign}", win, push=push, note=f"fav {fav}"))

    # --- Totales (medias: sin push) ---
    tnote = f"~{exp_goals:.1f} goles esperados"
    for line in _TOTAL_HALF:
        over = float(matrix[total > line].sum())
        out.append(_sel("Goles Más/Menos", f"Más de {line:g}", over, note=tnote))
        out.append(_sel("Goles Más/Menos", f"Menos de {line:g}", 1.0 - over, note=tnote))
    # --- Totales enteras (con push) ---
    for line in _TOTAL_WHOLE:
        over = float(matrix[total > line].sum())
        push = float(matrix[total == line].sum())
        under = float(matrix[total < line].sum())
        out.append(_sel("Goles asiáticos", f"Más de {line:g}", over, push=push, note=tnote))
        out.append(_sel("Goles asiáticos", f"Menos de {line:g}", under, push=push, note=tnote))

    # --- Totales por equipo ---
    home_dist = matrix.sum(axis=1)   # goles del local
    away_dist = matrix.sum(axis=0)   # goles de la visita
    for team, dist, xg in ((home, home_dist, home_xg), (away, away_dist, away_xg)):
        nt = f"~{xg:.1f} del equipo"
        for line in _TEAM_TOTAL:
            over = float(dist[idx > line].sum())
            out.append(_sel("Total por equipo", f"{team} Más de {line:g}", over, note=nt))
            out.append(_sel("Total por equipo", f"{team} Menos de {line:g}", 1.0 - over, note=nt))

    # --- Ambos marcan ---
    btts_yes = float(matrix[(i >= 1) & (j >= 1)].sum())
    out += [
        _sel("Ambos marcan", "Sí", btts_yes, note=tnote),
        _sel("Ambos marcan", "No", 1.0 - btts_yes, note=tnote),
    ]

    # --- Par / Impar (total de goles) ---
    even = float(matrix[(total % 2) == 0].sum())
    out += [
        _sel("Par/Impar", "Par", even),
        _sel("Par/Impar", "Impar", 1.0 - even),
    ]

    # --- Valla invicta / gana a cero ---
    cs_home = float(matrix[:, 0].sum())   # visita no marca
    cs_away = float(matrix[0, :].sum())   # local no marca
    wtn_home = float(matrix[(margin > 0) & (j == 0)].sum())
    wtn_away = float(matrix[(margin < 0) & (i == 0)].sum())
    out += [
        _sel("Valla invicta", f"{home} sin recibir", cs_home),
        _sel("Valla invicta", f"{away} sin recibir", cs_away),
        _sel("Gana a cero", f"{home} gana sin recibir", wtn_home),
        _sel("Gana a cero", f"{away} gana sin recibir", wtn_away),
    ]

    # --- Goles exactos (0,1,2,3,4,5+) ---
    for n_goals in range(5):
        out.append(_sel("Goles exactos", f"{n_goals}", float(matrix[total == n_goals].sum())))
    out.append(_sel("Goles exactos", "5+", float(matrix[total >= 5].sum())))

    # --- Marcador exacto (top 6) ---
    flat = [(matrix[a, b], a, b) for a in range(n) for b in range(n)]
    flat.sort(reverse=True)
    for prob, a, b in flat[:6]:
        out.append(_sel("Marcador exacto", f"{a}-{b}", float(prob)))

    return out


def _count_selections(
    label: str, total: float, variance: float, ladder: tuple[float, ...]
) -> list[Selection]:
    """Over/Under de un mercado de conteo (córners/tarjetas) en toda su escalera."""
    note = f"~{total:.1f} esperados"
    out: list[Selection] = []
    for line in ladder:
        over, under = over_under(total, line, variance=variance)
        out.append(_sel(label, f"Más de {line:g}", over, note=note))
        out.append(_sel(label, f"Menos de {line:g}", under, note=note))
    return out


def build_market_book(
    home: str,
    away: str,
    *,
    elo: EloModel,
    goals: GoalsModel | None,
    corners: CornersModel | None = None,
    cards: CardsModel | None = None,
    referee: str | None = None,
    knockout: bool = False,
    neutral: bool = True,
    match_name: str | None = None,
) -> MarketBook:
    """Arma el libro de mercados completo de un partido (todos los mercados + cuota justa)."""
    p = elo.predict(home, away, neutral=neutral)
    selections: list[Selection] = []
    home_xg = away_xg = 0.0
    dc_home = dc_draw = dc_away = 0.0

    if goals is not None and goals.can_predict(home, away):
        try:
            matrix, home_xg, away_xg = goals.score_matrix(home, away, neutral=neutral)
            selections += _goals_selections(matrix, home, away, home_xg, away_xg)
            n = matrix.shape[0]
            m = np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1)
            dc_home = float(matrix[m > 0].sum())
            dc_draw = float(matrix[m == 0].sum())
            dc_away = float(matrix[m < 0].sum())
        except GoalsModelError:
            pass

    if corners is not None:
        cp = corners.predict(home, away)
        selections += _count_selections(
            "Córners Más/Menos", cp.total, cp.total * corners.dispersion, _CORNER_LADDER
        )

    if cards is not None:
        cdp = cards.predict(home, away, referee=referee, knockout=knockout)
        variance = cdp.total * getattr(cards, "dispersion", 1.0)
        selections += _count_selections("Tarjetas Más/Menos", cdp.total, variance, _CARD_LADDER)

    return MarketBook(
        match=match_name or f"{home} vs {away}",
        home=home, away=away, home_xg=home_xg, away_xg=away_xg,
        p_home=p.home, p_draw=p.draw, p_away=p.away,
        exp_goals=home_xg + away_xg,
        dc_home=dc_home, dc_draw=dc_draw, dc_away=dc_away,
        selections=selections,
    )


def format_market_book(book: MarketBook, *, min_prob: float = 0.0) -> str:
    """Texto plano del libro completo (para que Claude razone). Cuota justa por selección."""
    head = (
        f"PANORAMA — {book.match}\n"
        f"1X2 (Elo, baseline ganador): {book.home} {book.p_home:.0%} / "
        f"X {book.p_draw:.0%} / {book.away} {book.p_away:.0%}\n"
        f"1X2 (Dixon-Coles, de goles): {book.home} {book.dc_home:.0%} / "
        f"X {book.dc_draw:.0%} / {book.away} {book.dc_away:.0%}\n"
        f"xG modelo: {book.home} {book.home_xg:.2f} − {book.away} {book.away_xg:.2f} "
        f"(total ~{book.exp_goals:.2f})\n"
        "Nota: si Elo y Dixon-Coles difieren mucho en el ganador, Elo manda en 1X2; "
        "DC manda en goles/hándicaps/totales. Las selecciones de abajo salen de la "
        "matriz de goles (DC).\n"
    )
    lines = [head]
    for market, sels in book.by_market().items():
        shown = [s for s in sels if s.prob >= min_prob]
        if not shown:
            continue
        lines.append(f"\n[{market}]")
        for s in shown:
            extra = f" · push {s.push:.0%}" if s.push > 0.01 else ""
            lines.append(f"  {s.pick}: {s.prob:.0%} (cuota s/modelo {s.fair:.2f}){extra}")
    return "\n".join(lines)
