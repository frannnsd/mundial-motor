"""Evaluador de cuotas: cuáles son BUENAS y qué combinadas valen.

Compara la probabilidad del modelo contra la cuota OFRECIDA. Una cuota es BUENA
cuando paga más de lo que el modelo dice que vale: EV = prob·cuota − 1 > 0.
No es "esta es la cuota justa" — es "esta cuota conviene jugarla".

Para combinadas: multiplica probabilidades y cuotas. Una combinada x1000 con chance
real (prob·cuota_combinada > 1) es buena para probar con poco — el evaluador la marca.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import prod

from mundial_bot.collectors.odds_af import MarketOdds
from mundial_bot.report import MatchReport

# Mapeo de los mercados del modelo a los nombres de API-Football.
_WINNER_OUTCOME = {"home": "Home", "draw": "Draw", "away": "Away"}

# Si el modelo difiere del mercado MÁS que esto, es error del modelo, no value.
# El mercado (14+ casas + Pinnacle) es muy difícil de superar; un "edge" enorme = bug.
MAX_MODEL_MARKET_GAP = 0.10
# Tope de EV creíble: arriba de esto casi siempre es error del modelo.
MAX_BELIEVABLE_EV = 0.25


@dataclass(frozen=True)
class GoodBet:
    match: str
    market: str       # "Ganador", "Goles", "Ambos marcan", "Córners", "Tarjetas"
    pick: str         # texto legible
    model_prob: float
    odds: float
    book: str

    @property
    def ev(self) -> float:
        return self.model_prob * self.odds - 1.0

    @property
    def implied(self) -> float:
        return 1.0 / self.odds


def _check(
    bets: list[GoodBet], match: str, market: str, pick: str, prob: float,
    market_odds: MarketOdds | None, outcome: str, *, min_ev: float,
) -> None:
    if market_odds is None:
        return
    found = market_odds.best.get(outcome)
    if not found:
        return
    odd, book = found
    bet = GoodBet(match, market, pick, prob, odd, book)
    gap = prob - (1.0 / odd)
    # Edge creíble: EV positivo pero no delirante, y el modelo no muy lejos del mercado.
    if min_ev <= bet.ev <= MAX_BELIEVABLE_EV and gap <= MAX_MODEL_MARKET_GAP:
        bets.append(bet)


def evaluate_match(
    report: MatchReport, odds: dict[str, MarketOdds], *, min_ev: float = 0.0
) -> list[GoodBet]:
    """Devuelve las apuestas BUENAS (cuota que paga más de lo que vale) de un partido."""
    bets: list[GoodBet] = []
    w = report.winner
    _check(bets, report.match, "Ganador", w.pick, w.prob,
           odds.get("Match Winner"), _WINNER_OUTCOME.get(w.side, ""), min_ev=min_ev)

    if report.goals:
        g = report.goals
        side = "Over" if g.side == "over" else "Under"
        _check(bets, report.match, "Goles", g.pick, g.prob,
               odds.get("Goals Over/Under"), f"{side} {g.line}", min_ev=min_ev)

    if report.btts:
        b = report.btts
        _check(bets, report.match, "Ambos marcan", b.pick, b.prob,
               odds.get("Both Teams Score"), "Yes" if b.side == "yes" else "No", min_ev=min_ev)

    return bets


@dataclass(frozen=True)
class GoodParlay:
    legs: tuple[GoodBet, ...]

    @property
    def combined_prob(self) -> float:
        return prod(leg.model_prob for leg in self.legs)

    @property
    def combined_odds(self) -> float:
        return prod(leg.odds for leg in self.legs)

    @property
    def ev(self) -> float:
        return self.combined_prob * self.combined_odds - 1.0


def build_parlays(
    bets: list[GoodBet], *, sizes: tuple[int, ...] = (2, 3, 4),
    min_ev: float = 0.0, max_results: int = 8,
) -> list[GoodParlay]:
    """Arma combinadas de patas de partidos distintos con EV combinado positivo.

    Incluye las de cuota alta (x100, x1000) si la chance combinada las hace +EV.
    """
    out: list[GoodParlay] = []
    for size in sizes:
        for combo in itertools.combinations(bets, size):
            if len({leg.match for leg in combo}) < size:
                continue  # patas del mismo partido → correlacionadas, se saltean
            parlay = GoodParlay(legs=tuple(combo))
            if parlay.ev >= min_ev:
                out.append(parlay)
    out.sort(key=lambda p: p.ev, reverse=True)
    return out[:max_results]


def format_good_bets(bets: list[GoodBet]) -> str:
    """Lista de cuotas buenas para Telegram (HTML), directo."""
    if not bets:
        return "Hoy no hay cuotas que paguen más de lo que valen. 🤷"
    bets = sorted(bets, key=lambda b: b.ev, reverse=True)
    lines = ["🔥 <b>CUOTAS BUENAS HOY</b> (pagan más de lo que valen)"]
    for b in bets:
        lines.append(
            f"⚽ {b.match}\n"
            f"   🎯 <b>{b.pick}</b> @ <b>{b.odds:.2f}</b> ({b.book})\n"
            f"   modelo {b.model_prob:.0%} vs paga {b.implied:.0%} → "
            f"EV <b>+{b.ev:.0%}</b>"
        )
    return "\n".join(lines)


def format_parlays(parlays: list[GoodParlay]) -> str:
    """Lista de combinadas que valen (incluidas las de cuota alta)."""
    if not parlays:
        return ""
    lines = ["", "🎲 <b>COMBINADAS QUE VALEN</b>"]
    for p in parlays:
        legs = " + ".join(leg.pick for leg in p.legs)
        lines.append(
            f"   <b>{p.combined_odds:.2f}x</b> · chance {p.combined_prob:.1%} · "
            f"EV <b>+{p.ev:.0%}</b>\n   {legs}"
        )
    return "\n".join(lines)


def format_evaluation(bets: list[GoodBet], parlays: list[GoodParlay], *, date_str: str) -> str:
    """Mensaje completo: cuotas buenas + combinadas."""
    return (
        f"📅 <b>{date_str}</b>\n\n"
        + format_good_bets(bets)
        + format_parlays(parlays)
    )
