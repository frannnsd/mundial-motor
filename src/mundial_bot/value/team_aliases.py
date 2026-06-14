"""Normalización de nombres de equipos entre The Odds API y el dataset martj42.

Las casas y el dataset histórico a veces nombran distinto a la misma selección
(ej. "USA" vs "United States"). Sin esto, el modelo no encuentra el rating del
equipo y el partido se descarta. El mapa es extensible: cuando lleguen las cuotas
reales, `unmatched_teams()` ayuda a detectar los que falten para agregarlos.
"""

from __future__ import annotations

# Nombre como lo da API-Football  ->  nombre como está en martj42 (modelo Elo).
# Verificado contra los 48 equipos del Mundial 2026 (solo estos 5 difieren).
ALIASES: dict[str, str] = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "USA": "United States",
    # Nota: martj42 ya usa "Ivory Coast" y "South Korea" → NO se remapean.
}


def normalize_team(name: str) -> str:
    """Devuelve el nombre canónico (martj42) de un equipo."""
    return ALIASES.get(name, name)


def unmatched_teams(odds_teams: set[str], known_teams: set[str]) -> set[str]:
    """Equipos de las cuotas que, ya normalizados, no existen en el modelo.

    Útil para detectar aliases faltantes cuando llegan cuotas reales.
    """
    return {t for t in odds_teams if normalize_team(t) not in known_teams}
