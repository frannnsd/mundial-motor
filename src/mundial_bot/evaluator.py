"""Escáner de partidos: lo MÁS PROBABLE de cada partido + la cuota que paga.

Sin "value": NO exige que la cuota pague más de lo justo. Franco decide. El bot solo
muestra, por mercado, la jugada más probable según el modelo (con su probabilidad) y la
cuota que ofrece la casa — y arma combinadas por su CHANCE combinada (las más probables
y las de mayor pago), sin descartar nada por no tener "edge".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import prod

from mundial_bot.collectors.odds_af import MarketOdds
from mundial_bot.report import MatchReport

# Lado del modelo → nombre del resultado en las casas (API-Football / odds-api.io).
_WINNER_OUTCOME = {"home": "Home", "draw": "Draw", "away": "Away"}
# Posibles nombres del mercado de córners/tarjetas según la casa (probamos varios).
_CORNER_MARKETS = ("Corners Over/Under", "Total Corners", "Corners Over Under")
_CARD_MARKETS = ("Cards Over/Under", "Total Cards", "Cards Over Under")


@dataclass(frozen=True)
class Play:
    """Una jugada probable de un mercado, con la cuota que paga la casa (si la lista)."""

    match: str
    market: str        # "Ganador", "Goles", "Ambos marcan", "Córners", "Tarjetas"
    pick: str
    prob: float        # probabilidad del modelo
    odd: float | None = None   # cuota de la casa (None si no está listada)
    book: str = ""

    @property
    def implied(self) -> float | None:
        """Probabilidad implícita de la cuota de la casa (para mostrar al lado del modelo)."""
        return (1.0 / self.odd) if self.odd else None

    @property
    def model_odds(self) -> float:
        """Cuota que correspondería a la probabilidad del modelo."""
        return round(1.0 / self.prob, 2) if self.prob > 0 else 0.0


def _odd(odds: dict[str, MarketOdds], markets, outcome: str) -> tuple[float, str] | None:
    """Mejor cuota de un resultado probando varios nombres de mercado."""
    for market in markets:
        mo = odds.get(market)
        if mo and outcome in mo.best:
            return mo.best[outcome]
    return None


def scan_match(report: MatchReport, odds: dict[str, MarketOdds]) -> list[Play]:
    """Jugada más probable de cada mercado del partido, con la cuota que paga la casa."""
    plays: list[Play] = []

    def add(market: str, pick, markets, outcome: str) -> None:
        if pick is None:
            return
        found = _odd(odds, markets, outcome)
        odd, book = found if found else (None, "")
        plays.append(Play(report.match, market, pick.pick, pick.prob, odd, book))

    w = report.winner
    add("Ganador", w, ("Match Winner",), _WINNER_OUTCOME.get(w.side, ""))
    if report.goals:
        side = "Over" if report.goals.side == "over" else "Under"
        add("Goles", report.goals, ("Goals Over/Under",), f"{side} {report.goals.line}")
    if report.btts:
        add("Ambos marcan", report.btts, ("Both Teams Score",),
            "Yes" if report.btts.side == "yes" else "No")
    if report.corners:
        side = "Over" if report.corners.side == "over" else "Under"
        add("Córners", report.corners, _CORNER_MARKETS, f"{side} {report.corners.line}")
    if report.cards:
        side = "Over" if report.cards.side == "over" else "Under"
        add("Tarjetas", report.cards, _CARD_MARKETS, f"{side} {report.cards.line}")
    return plays


@dataclass(frozen=True)
class Combo:
    """Combinada: varias patas de partidos distintos, evaluada por su chance combinada."""

    legs: tuple[Play, ...]

    @property
    def combined_prob(self) -> float:
        return prod(leg.prob for leg in self.legs)

    @property
    def combined_odds(self) -> float:
        return prod(leg.odd for leg in self.legs if leg.odd)


def build_combos(
    plays: list[Play], *, sizes: tuple[int, ...] = (2, 3, 4),
    top_likely: int = 4, top_payout: int = 4,
) -> tuple[list[Combo], list[Combo]]:
    """Combinadas de patas con cuota real, de partidos distintos.

    Devuelve (más_probables, de_mayor_pago). Sin filtro de value: las muestra por chance.
    """
    # Dedup (una jugada puede venir en varias listas) y solo patas con cuota real.
    uniq: dict[tuple[str, str, str], Play] = {}
    for p in plays:
        if p.odd:
            uniq.setdefault((p.match, p.market, p.pick), p)
    legs = list(uniq.values())
    combos: list[Combo] = []
    for size in sizes:
        for combo in itertools.combinations(legs, size):
            if len({leg.match for leg in combo}) < size:
                continue  # patas del mismo partido → correlacionadas, se saltean
            combos.append(Combo(legs=tuple(combo)))
    likely = sorted(combos, key=lambda c: c.combined_prob, reverse=True)[:top_likely]
    payout = sorted(combos, key=lambda c: c.combined_odds, reverse=True)[:top_payout]
    return likely, payout


def _pct(x: float) -> str:
    return f"{x:.0%}"


def format_play(p: Play) -> str:
    """Una jugada: lo más probable + lo que paga (sin lenguaje de value)."""
    if p.odd:
        return (
            f"   🎯 <b>{p.pick}</b> — modelo {_pct(p.prob)} · paga <b>{p.odd:.2f}</b> "
            f"({p.book}, implícita {_pct(p.implied)})"
        )
    return f"   🎯 <b>{p.pick}</b> — modelo {_pct(p.prob)} · cuota del modelo ~{p.model_odds:.2f}"


def format_combo(c: Combo) -> str:
    legs = " + ".join(leg.pick for leg in c.legs)
    return f"   <b>{c.combined_odds:.2f}x</b> · chance {c.combined_prob:.1%}\n   {legs}"


def plays_from_book(book, odds: dict[str, MarketOdds]) -> list[Play]:
    """Convierte TODAS las selecciones del libro que tengan cuota real en Play."""
    from mundial_bot.models.market_book import real_odd

    out: list[Play] = []
    for s in book.selections:
        found = real_odd(s, odds)
        if found:
            out.append(Play(book.match, s.market, s.pick, s.prob, found[0], found[1]))
    return out


def _cap_per_match(plays_sorted: list[Play], *, per_match: int = 2, total: int = 8) -> list[Play]:
    """Toma las mejores jugadas pero sin saturar con un solo partido."""
    seen: dict[str, int] = {}
    out: list[Play] = []
    for p in plays_sorted:
        if seen.get(p.match, 0) >= per_match:
            continue
        seen[p.match] = seen.get(p.match, 0) + 1
        out.append(p)
        if len(out) >= total:
            break
    return out


# Umbrales del escaneo (no son "value gatekeeping": solo ordenan qué mostrar primero).
_FIRME_PROB = 0.62          # firme = el modelo la ve clara
_FIRME_MIN_ODD = 1.40       # y que pague algo (no un 1.05 trivial)
_EDGE_MIN = 0.03            # el modelo la ve un poco mejor que la casa
_EDGE_MAX = 0.20            # gap creíble (más que esto suele ser error del modelo)
_BATACAZO_ODD = 5.0
_BATACAZO_PROB = (0.06, 0.40)   # batacazo = poco probable pero no imposible
# Si el modelo se desvía MÁS que esto de la casa, la cuota casi seguro está mal/stale
# (cuota vieja de un partido terminado, o mal mapeada). NO es value: es dato roto.
_MAX_GAP = 0.30


def best_plays(plays: list[Play]) -> tuple[list[Play], list[Play], list[Play]]:
    """Ordena TODAS las jugadas con cuota en: (más firmes, modelo>casa, batacazos).

    Descarta cuotas con un desvío imposible respecto del modelo (dato roto/stale), no
    por "value" sino por sanidad de los datos.
    """
    priced = [
        p for p in plays
        if p.odd and p.implied is not None and (p.prob - p.implied) <= _MAX_GAP
    ]
    firmes = _cap_per_match(
        sorted([p for p in priced if p.prob >= _FIRME_PROB and p.odd >= _FIRME_MIN_ODD],
               key=lambda p: -p.prob)
    )
    mejor = _cap_per_match(
        sorted([p for p in priced
                if _EDGE_MIN <= (p.prob - p.implied) <= _EDGE_MAX and 1.4 <= p.odd <= 8.0],
               key=lambda p: -(p.prob - p.implied))
    )
    lo, hi = _BATACAZO_PROB
    batacazos = _cap_per_match(
        sorted([p for p in priced if p.odd >= _BATACAZO_ODD and lo <= p.prob <= hi],
               key=lambda p: -p.prob),
        per_match=1, total=5,
    )
    return firmes, mejor, batacazos


def _play_line(p: Play) -> str:
    extra = f", casa {p.implied:.0%}" if p.implied is not None else ""
    return (
        f"   ⚽ {p.match}\n"
        f"   🎯 <b>{p.pick}</b> — modelo {p.prob:.0%} · paga <b>{p.odd:.2f}</b> ({p.book}{extra})"
    )


def format_full_scan(
    firmes: list[Play], mejor: list[Play], batacazos: list[Play],
    likely: list[Combo], payout: list[Combo], *, date_str: str,
) -> str:
    """Escaneo de TODOS los mercados: las mejores jugadas del día + combinadas."""
    if not (firmes or mejor or batacazos):
        return f"🔍 <b>ESCANEO DEL DÍA — {date_str}</b>\n\nSin cuotas para escanear ahora. 🤷"

    lines = [
        f"🔍 <b>ESCANEO DEL DÍA — {date_str}</b>",
        "(todos los mercados · modelo vs lo que paga)",
    ]

    def block(title: str, plays: list[Play]) -> None:
        if not plays:
            return
        lines.append(f"\n{title}")
        lines.extend(_play_line(p) for p in plays)

    block("🔒 <b>LAS MÁS FIRMES</b>", firmes)
    block("🎯 <b>EL MODELO LAS VE MEJOR QUE LA CASA</b>", mejor)
    block("🚀 <b>BATACAZOS</b> (poco probables, pagan fuerte)", batacazos)
    if likely:
        lines.append("\n🎲 <b>COMBINADAS MÁS PROBABLES</b>")
        lines.extend(format_combo(c) for c in likely)
    if payout:
        lines.append("\n💰 <b>COMBINADAS DE MAYOR PAGO</b> (más arriesgadas, vos ves)")
        lines.extend(format_combo(c) for c in payout)
    return "\n".join(lines)


def format_scan(plays: list[Play], *, date_str: str) -> str:
    """Escaneo del día: por partido, lo más probable + la cuota; después combinadas."""
    if not plays:
        return f"📅 <b>{date_str}</b>\n\nNo tengo partidos para escanear ahora. 🤷"

    by_match: dict[str, list[Play]] = {}
    for p in plays:
        by_match.setdefault(p.match, []).append(p)

    lines = [f"🔍 <b>ESCANEO DEL DÍA — {date_str}</b>", "(lo más probable + lo que paga)"]
    for match, mplays in by_match.items():
        lines.append(f"\n⚽ <b>{match}</b>")
        lines += [format_play(p) for p in mplays]

    likely, payout = build_combos(plays)
    if likely:
        lines.append("\n🎲 <b>COMBINADAS MÁS PROBABLES</b>")
        lines += [format_combo(c) for c in likely]
    if payout:
        lines.append("\n💰 <b>COMBINADAS DE MAYOR PAGO</b> (más arriesgadas, vos ves)")
        lines += [format_combo(c) for c in payout]
    return "\n".join(lines)
