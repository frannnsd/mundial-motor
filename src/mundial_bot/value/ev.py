"""Valor esperado (EV) y detección de value bets.

El edge de todo el sistema vive acá: una apuesta es de **valor** cuando la
probabilidad del modelo supera la implícita de la cuota, es decir cuando
`EV = p·cuota − 1 > 0` (equivalente a `p > 1/cuota`).

Solo marcamos picks cuyo edge supera un umbral mínimo (`min_edge`), para no
apostar sobre ruido cuando el borde es marginal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Selection:
    """Una opción apostable concreta ofrecida por una casa."""

    match: str        # "Argentina vs Mexico"
    market: str       # "1X2", "OU2.5", "BTTS"
    selection: str    # "home", "draw", "away", "over", "under", "yes", "no"
    odds: float       # cuota decimal ofrecida
    bookmaker: str = "?"

    @property
    def label(self) -> str:
        return f"{self.match} | {self.market}:{self.selection} @ {self.odds:.2f}"


@dataclass(frozen=True)
class ValuePick:
    """Una selección evaluada contra la probabilidad del modelo."""

    selection: Selection
    model_prob: float            # probabilidad estimada por nuestro modelo
    edge: float                  # EV por unidad apostada = model_prob·odds − 1
    fair_prob: float | None = None  # prob. justa del mercado (de-vig), para comparar/CLV
    meta: dict = field(default_factory=dict)

    @property
    def ev(self) -> float:
        """Alias semántico: el EV por unidad coincide con el edge."""
        return self.edge

    @property
    def is_value(self) -> bool:
        return self.edge > 0

    @property
    def model_vs_market(self) -> float | None:
        """Cuánto más probable lo ve el modelo vs el mercado (diferencia de prob.)."""
        if self.fair_prob is None:
            return None
        return self.model_prob - self.fair_prob


def expected_value(prob: float, odds: float) -> float:
    """EV por unidad apostada: `p·cuota − 1`. Positivo ⇒ apuesta de valor."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"La probabilidad debe estar en [0,1], recibí {prob}")
    if odds <= 1.0:
        raise ValueError(f"La cuota decimal debe ser > 1.0, recibí {odds}")
    return prob * odds - 1.0


def evaluate(
    selection: Selection, model_prob: float, *, fair_prob: float | None = None
) -> ValuePick:
    """Evalúa una selección con la probabilidad del modelo → ValuePick."""
    edge = expected_value(model_prob, selection.odds)
    return ValuePick(
        selection=selection, model_prob=model_prob, edge=edge, fair_prob=fair_prob
    )


def find_value_bets(
    candidates: list[tuple[Selection, float]],
    *,
    min_edge: float = 0.03,
    fair_probs: dict[str, float] | None = None,
) -> list[ValuePick]:
    """Filtra y ordena las apuestas de valor de una lista de candidatos.

    Args:
        candidates: lista de (Selection, probabilidad_del_modelo).
        min_edge: edge mínimo para considerar valor (ej. 0.03 = +3% EV).
        fair_probs: opcional, mapa Selection.label -> prob. justa del mercado.

    Returns:
        Lista de ValuePick con edge >= min_edge, ordenada de mayor a menor edge.
    """
    picks: list[ValuePick] = []
    for sel, model_prob in candidates:
        fair = fair_probs.get(sel.label) if fair_probs else None
        pick = evaluate(sel, model_prob, fair_prob=fair)
        if pick.edge >= min_edge:
            picks.append(pick)
    picks.sort(key=lambda p: p.edge, reverse=True)
    return picks
