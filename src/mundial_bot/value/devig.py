"""De-vig: quita el margen de la casa de las cuotas para obtener probabilidades justas.

La cuota cruda implica `1/cuota`, pero la suma de las implícitas de un mercado es
> 1 (ese exceso es el margen / overround de la casa). El de-vig lo remueve.

Usamos el método **Shin** por defecto (modela la fracción de apostadores informados
y es el mejor calibrado en mercados líquidos), vía la implementación de penaltyblog.
La probabilidad justa resultante es nuestro mejor proxy de la "probabilidad real de
mercado" — el termómetro contra el que se compara la probabilidad del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

from penaltyblog.implied import ImpliedMethod, calculate_implied

# Métodos de de-vig soportados (penaltyblog).
SUPPORTED_METHODS = {m.value for m in ImpliedMethod}


@dataclass(frozen=True)
class FairProbabilities:
    """Probabilidades justas de un mercado tras quitar el margen."""

    outcomes: dict[str, float]  # nombre del resultado -> probabilidad justa
    margin: float               # margen de la casa removido (overround - 1)
    method: str                 # método de de-vig usado

    def __getitem__(self, name: str) -> float:
        return self.outcomes[name]


def decimal_to_implied(odds: float) -> float:
    """Probabilidad implícita cruda de una cuota decimal (sin quitar margen)."""
    if odds <= 1.0:
        raise ValueError(f"La cuota decimal debe ser > 1.0, recibí {odds}")
    return 1.0 / odds


def overround(odds_by_outcome: dict[str, float]) -> float:
    """Suma de las implícitas crudas menos 1 (el margen bruto del mercado)."""
    return sum(decimal_to_implied(o) for o in odds_by_outcome.values()) - 1.0


def devig(odds_by_outcome: dict[str, float], method: str = "shin") -> FairProbabilities:
    """Convierte un mercado de cuotas decimales en probabilidades justas.

    Args:
        odds_by_outcome: mapa resultado -> cuota decimal (ej. {"home":2.1,"draw":3.4,"away":3.8}).
        method: "shin" (default), "power", "multiplicative", etc.

    Returns:
        FairProbabilities con probabilidades que suman 1 y el margen removido.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Método de de-vig '{method}' no soportado. Opciones: {SUPPORTED_METHODS}")
    if len(odds_by_outcome) < 2:
        raise ValueError("Un mercado necesita al menos 2 resultados para de-vig.")

    names = list(odds_by_outcome.keys())
    values = [float(odds_by_outcome[n]) for n in names]

    result = calculate_implied(values, method=method, market_names=names)
    fair = {n: float(p) for n, p in zip(names, result.probabilities, strict=True)}
    return FairProbabilities(outcomes=fair, margin=float(result.margin), method=method)
