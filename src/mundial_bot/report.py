"""Reporte de predicción multi-mercado por partido.

Para cada partido (ej. "Marruecos vs Brasil") muestra lo más probable en cada
mercado — ganador, goles, córners, tarjetas, ambos marcan — con su **cuota justa**
(= 1 / probabilidad). Vos comparás esa cuota justa contra lo que paga tu casa
(Bet365/bplay/Stake): si paga más, hay value.

Combina cuatro modelos: Elo (ganador), Dixon-Coles (goles/BTTS), córners y tarjetas.
Cada modelo es opcional: si falta, ese mercado simplemente no aparece.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.models.count_market import closest_line
from mundial_bot.models.elo_model import EloModel
from mundial_bot.models.goals_model import GoalsModel, GoalsModelError

GOAL_LINES = (1.5, 2.5, 3.5)


@dataclass(frozen=True)
class MarketPick:
    """La opción más probable de un mercado, con su cuota justa."""

    pick: str                      # "Brasil", "Under 2.5 goles", "Over 9.5 córners"
    prob: float
    expected: float | None = None  # valor esperado (goles/córners/tarjetas)

    @property
    def fair_odds(self) -> float:
        return round(1.0 / self.prob, 2) if self.prob > 0 else 0.0


@dataclass(frozen=True)
class MatchReport:
    match: str
    home_prob: float
    draw_prob: float
    away_prob: float
    winner: MarketPick
    goals: MarketPick | None = None
    btts: MarketPick | None = None
    corners: MarketPick | None = None
    cards: MarketPick | None = None


def _favored(over_label: str, p_over: float, under_label: str, p_under: float,
             *, expected: float | None = None) -> MarketPick:
    if p_over >= p_under:
        return MarketPick(over_label, p_over, expected)
    return MarketPick(under_label, p_under, expected)


def build_match_report(
    home: str,
    away: str,
    *,
    elo: EloModel,
    goals: GoalsModel | None = None,
    corners: CornersModel | None = None,
    cards: CardsModel | None = None,
    referee: str | None = None,
    knockout: bool = False,
    neutral: bool = True,
    match_name: str | None = None,
) -> MatchReport:
    """Arma el reporte multi-mercado de un partido."""
    p = elo.predict(home, away, neutral=neutral)
    home_name, away_name = home, away

    # Ganador más probable.
    sides = [(home_name, p.home), ("Empate", p.draw), (away_name, p.away)]
    win_label, win_prob = max(sides, key=lambda s: s[1])
    winner = MarketPick(win_label, win_prob)

    goals_pick = btts_pick = None
    if goals is not None and goals.can_predict(home, away):
        try:
            m = goals.predict(home, away, neutral=neutral)
            line = closest_line(m.exp_goals, GOAL_LINES)
            over, under = m.lines.get(line, (m.over_2_5, m.under_2_5))
            goals_pick = _favored(
                f"Over {line} goles", over, f"Under {line} goles", under,
                expected=m.exp_goals,
            )
            btts_pick = _favored(
                "Ambos marcan: Sí", m.btts_yes, "Ambos marcan: No", m.btts_no
            )
        except GoalsModelError:
            pass

    corners_pick = None
    if corners is not None:
        cp = corners.predict(home, away)
        corners_pick = _favored(
            f"Over {cp.line} córners", cp.p_over, f"Under {cp.line} córners", cp.p_under,
            expected=cp.total,
        )

    cards_pick = None
    if cards is not None:
        cdp = cards.predict(home, away, referee=referee, knockout=knockout)
        cards_pick = _favored(
            f"Over {cdp.line} tarjetas", cdp.p_over, f"Under {cdp.line} tarjetas", cdp.p_under,
            expected=cdp.total,
        )

    return MatchReport(
        match=match_name or f"{home} vs {away}",
        home_prob=p.home, draw_prob=p.draw, away_prob=p.away,
        winner=winner, goals=goals_pick, btts=btts_pick,
        corners=corners_pick, cards=cards_pick,
    )


def format_match_report(r: MatchReport) -> str:
    """Formatea un reporte de partido para Telegram (HTML)."""
    raw_home, raw_away = (r.match.split(" vs ", 1) + [""])[:2]
    match = html.escape(r.match)
    home, away = html.escape(raw_home), html.escape(raw_away)
    winner = html.escape(r.winner.pick)
    lines = [
        f"⚽ <b>{match}</b>",
        f"   🏆 Gana: <b>{winner}</b> ({r.winner.prob:.0%})"
        f" · justo @ {r.winner.fair_odds:.2f}",
        f"      [{home} {r.home_prob:.0%} · X {r.draw_prob:.0%}"
        f" · {away} {r.away_prob:.0%}]",
    ]
    if r.goals:
        lines.append(
            f"   ⚽ Goles: ~{r.goals.expected:.1f} → <b>{r.goals.pick}</b> "
            f"({r.goals.prob:.0%}) · justo @ {r.goals.fair_odds:.2f}"
        )
    if r.corners:
        lines.append(
            f"   🚩 Córners: ~{r.corners.expected:.1f} → <b>{r.corners.pick}</b> "
            f"({r.corners.prob:.0%}) · justo @ {r.corners.fair_odds:.2f}"
        )
    if r.cards:
        lines.append(
            f"   🟨 Tarjetas: ~{r.cards.expected:.1f} → <b>{r.cards.pick}</b> "
            f"({r.cards.prob:.0%}) · justo @ {r.cards.fair_odds:.2f}"
        )
    if r.btts:
        lines.append(
            f"   🤝 {r.btts.pick} ({r.btts.prob:.0%}) · justo @ {r.btts.fair_odds:.2f}"
        )
    return "\n".join(lines)


def format_match_reports(reports: list[MatchReport], *, date_str: str) -> str:
    """Cartilla de predicciones multi-mercado de varios partidos."""
    out = [f"🔮 <b>PREDICCIONES POR PARTIDO — {date_str}</b>", ""]
    for r in reports:
        out.append(format_match_report(r))
        out.append("")
    out.append(
        "ℹ️ <i>'Justo @' = cuota justa del modelo. Si tu casa (Bet365/bplay/Stake) "
        "paga MÁS, hay value. El bot sugiere; vos decidís.</i>"
    )
    return "\n".join(out)
