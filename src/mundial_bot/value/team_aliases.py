"""Normalización de nombres de equipos entre The Odds API y el dataset martj42.

Las casas y el dataset histórico a veces nombran distinto a la misma selección
(ej. "USA" vs "United States"). Sin esto, el modelo no encuentra el rating del
equipo y el partido se descarta. El mapa es extensible: cuando lleguen las cuotas
reales, `unmatched_teams()` ayuda a detectar los que falten para agregarlos.
"""

from __future__ import annotations

# Nombre como lo da The Odds API  ->  nombre como está en martj42.
ALIASES: dict[str, str] = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cape Verde Islands",
    "Czech Republic": "Czechia",
    "DR Congo": "DR Congo",
    "Republic of Ireland": "Ireland",
    "Curacao": "Curaçao",
    "Turkey": "Türkiye",
}


def normalize_team(name: str) -> str:
    """Devuelve el nombre canónico (martj42) de un equipo."""
    return ALIASES.get(name, name)


def unmatched_teams(odds_teams: set[str], known_teams: set[str]) -> set[str]:
    """Equipos de las cuotas que, ya normalizados, no existen en el modelo.

    Útil para detectar aliases faltantes cuando llegan cuotas reales.
    """
    return {t for t in odds_teams if normalize_team(t) not in known_teams}
