"""Motor de contexto: ajusta las tasas base del partido antes de simular.

Cada factor es un multiplicador chico y explicable. Por defecto todo es 1.0
(neutral) → la simulación base coincide con el modelo validado. El contexto solo
mueve la aguja cuando hay información real: estar por quedar afuera (motivación),
eliminación directa, rivalidad/clásico, calor o altura de la sede.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Multiplicador de ataque según la situación del equipo en la tabla.
MOTIVATION = {
    "must_win": 1.08,     # debe ganar sí o sí (juega a todo o nada)
    "normal": 1.0,
    "qualified": 0.96,    # ya clasificado, administra
    "dead_rubber": 0.93,  # intrascendente, sin nada en juego
}

MOTIVATION_LABELS = {
    "must_win": "debe ganar",
    "normal": "normal",
    "qualified": "ya clasificado",
    "dead_rubber": "intrascendente",
}


@dataclass
class MatchContext:
    """Multiplicadores que ajustan la simulación. 1.0 = neutral (sin efecto)."""

    home_attack: float = 1.0   # sobre el xG del local
    away_attack: float = 1.0   # sobre el xG del visitante
    cards_mult: float = 1.0    # tarjetas (rivalidad, importancia, knockout)
    tempo_mult: float = 1.0    # córners y tiros (intensidad/ritmo)
    late_damping: float = 1.0  # goles en el tramo final (calor/altura)
    knockout: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def neutral(self) -> bool:
        return (
            self.home_attack == 1.0
            and self.away_attack == 1.0
            and self.cards_mult == 1.0
            and self.tempo_mult == 1.0
            and self.late_damping == 1.0
        )


# Sedes del Mundial 2026 con calor fuerte (verano en EEUU/México) y con altura.
HEAT_CITIES = {
    "dallas", "arlington", "houston", "miami", "miami gardens", "atlanta", "kansas city",
    "monterrey", "guadalajara", "zapopan", "mexico city",
}
ALTITUDE_CITIES = {"mexico city", "guadalajara", "zapopan", "toluca"}

# Rivalidades clásicas (más tarjetas). Conjuntos de nombres del modelo (en inglés).
RIVALRIES = [
    frozenset(p) for p in (
        {"Argentina", "Brazil"}, {"Argentina", "England"}, {"Argentina", "Uruguay"},
        {"England", "Germany"}, {"Germany", "Netherlands"}, {"Spain", "Portugal"},
        {"Mexico", "United States"}, {"Brazil", "Argentina"}, {"Serbia", "Croatia"},
    )
]


def auto_context(
    *,
    home: str,
    away: str,
    knockout: bool = False,
    city: str | None = None,
    home_motivation: str = "normal",
    away_motivation: str = "normal",
) -> MatchContext:
    """Deriva el contexto SOLO a partir de los datos del partido (sin que el usuario elija).

    - knockout: viene de la ronda.
    - calor/altura: de la ciudad de la sede.
    - rivalidad: de los nombres (clásicos conocidos).
    - motivación: la calcula el caller desde la tabla (puntos) y se pasa acá.
    """
    c = (city or "").strip().lower()
    return build_context(
        knockout=knockout,
        rivalry=frozenset({home, away}) in RIVALRIES,
        heat=c in HEAT_CITIES,
        altitude=c in ALTITUDE_CITIES,
        home_motivation=home_motivation,
        away_motivation=away_motivation,
    )


def build_context(
    *,
    knockout: bool = False,
    rivalry: bool = False,
    heat: bool = False,
    altitude: bool = False,
    home_motivation: str = "normal",
    away_motivation: str = "normal",
) -> MatchContext:
    """Arma el contexto a partir de las perillas. Devuelve multiplicadores + notas."""
    ctx = MatchContext(knockout=knockout)
    ctx.home_attack *= MOTIVATION.get(home_motivation, 1.0)
    ctx.away_attack *= MOTIVATION.get(away_motivation, 1.0)

    if home_motivation == "must_win" or away_motivation == "must_win":
        ctx.cards_mult *= 1.05
        ctx.notes.append("Alguien se juega la vida: partido más intenso, +tarjetas.")
    if knockout:
        ctx.cards_mult *= 1.10
        ctx.notes.append("Eliminación directa: más fricción; si empatan hay alargue y penales.")
    if rivalry:
        ctx.cards_mult *= 1.12
        ctx.notes.append("Clásico/rivalidad: suele calentarse, +tarjetas.")
    if heat:
        ctx.late_damping *= 0.92
        ctx.tempo_mult *= 0.97
        ctx.notes.append("Calor: baja el ritmo y caen los goles sobre el final.")
    if altitude:
        ctx.late_damping *= 0.95
        ctx.notes.append("Altura: desgaste físico en el tramo final.")
    return ctx
