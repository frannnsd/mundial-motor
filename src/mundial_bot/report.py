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
    sides = [(f"Gana {home_name}", p.home), ("Empate", p.draw), (f"Gana {away_name}", p.away)]
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


def _confidence(prob: float) -> str:
    """Etiqueta de confianza según la probabilidad."""
    if prob >= 0.70:
        return "🔥 Alta"
    if prob >= 0.62:
        return "💪 Media-alta"
    if prob >= 0.55:
        return "👍 Media"
    return "🤏 Pareja (mejor evitar)"


def best_bet(r: MatchReport) -> MarketPick:
    """La apuesta más firme del partido = la de mayor probabilidad."""
    picks = [r.winner]
    picks += [m for m in (r.goals, r.corners, r.cards, r.btts) if m is not None]
    return max(picks, key=lambda pk: pk.prob)


def format_match_report(r: MatchReport) -> str:
    """Reporte de un partido para Telegram: a qué apostar + probabilidad."""
    raw_home, raw_away = (r.match.split(" vs ", 1) + [""])[:2]
    match = html.escape(r.match)
    home, away = html.escape(raw_home), html.escape(raw_away)
    winner = html.escape(r.winner.pick)

    lines = [
        f"⚽ <b>{match}</b>",
        f"   🏆 <b>{winner}</b> — {r.winner.prob:.0%}",
        f"      [{home} {r.home_prob:.0%} · X {r.draw_prob:.0%} · {away} {r.away_prob:.0%}]",
    ]
    if r.goals:
        lines.append(
            f"   ⚽ {html.escape(r.goals.pick)} — {r.goals.prob:.0%}"
            f"  (esperados ~{r.goals.expected:.1f})"
        )
    if r.corners:
        lines.append(
            f"   🚩 {html.escape(r.corners.pick)} — {r.corners.prob:.0%}"
            f"  (~{r.corners.expected:.0f})"
        )
    if r.cards:
        lines.append(
            f"   🟨 {html.escape(r.cards.pick)} — {r.cards.prob:.0%}"
            f"  (~{r.cards.expected:.1f})"
        )
    if r.btts:
        lines.append(f"   🤝 {html.escape(r.btts.pick)} — {r.btts.prob:.0%}")

    bb = best_bet(r)
    lines.append(
        f"   ⭐ <b>Más firme:</b> {html.escape(bb.pick)} ({bb.prob:.0%}) · {_confidence(bb.prob)}"
    )
    return "\n".join(lines)


def format_match_reports(reports: list[MatchReport], *, date_str: str) -> str:
    """Cartilla de predicciones multi-mercado de varios partidos."""
    out = [f"🔮 <b>QUÉ APOSTAR HOY — {date_str}</b>", ""]
    for r in reports:
        out.append(format_match_report(r))
        out.append("")
    out.append(
        "ℹ️ <i>Probabilidades de nuestro modelo. El bot sugiere a qué apostar; "
        "vos decidís. Apostá con responsabilidad.</i>"
    )
    return "\n".join(out)
