"""Capa de props por JUGADOR (Fase B): reparte lo que el equipo predice.

El modelo de jugador NO compite con el de equipo — REPARTE su total:
P(cantidad de jugador) = share histórico × total del equipo × (minutos/90),
con dispersión Poisson por jugador. Ver `shares.py` (shares con shrinkage por
puesto) y `props.py` (minutos esperados + props coherentes con el equipo).
"""

from mundial_bot.players.props import expected_minutes, match_props
from mundial_bot.players.shares import player_shares

__all__ = ["expected_minutes", "match_props", "player_shares"]
