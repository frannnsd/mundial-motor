"""Forward test de props por jugador: log de predicciones + liquidación real.

Registra cada predicción ANTES del partido en SQLite y la liquida DESPUÉS con
las stats reales del fixture (mismo cache de `collectors.players_wc`, cero
llamadas extra si el JSON ya está en disco). Ver `log.py`.
"""

from mundial_bot.forward_test.log import log_prediction, settle_fixture, summary

__all__ = ["log_prediction", "settle_fixture", "summary"]
