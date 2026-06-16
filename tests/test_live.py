"""Tests del modo EN VIVO (matriz de marcador final desde el estado actual)."""

from __future__ import annotations

import numpy as np
import pytest

from mundial_bot.models.live import live_final_matrix


def _p_home(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    m = np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1)
    return float(matrix[m > 0].sum())


def test_live_matrix_suma_uno_y_respeta_el_marcador():
    matrix = live_final_matrix(1.5, 1.2, home_goals=2, away_goals=1, minute=60)
    assert matrix.sum() == pytest.approx(1.0)
    # El marcador final no puede ser menor al actual.
    assert matrix[:2, :].sum() == pytest.approx(0.0)   # local con <2 goles: imposible
    assert matrix[:, :1].sum() == pytest.approx(0.0)   # visita con <1 gol: imposible


def test_live_minuto_90_concentra_en_el_marcador_actual():
    matrix = live_final_matrix(2.0, 2.0, home_goals=2, away_goals=1, minute=90)
    assert matrix[2, 1] == pytest.approx(1.0, abs=1e-6)


def test_live_ganando_sube_la_prob_de_ganar():
    # 2-0 a los 80' → casi seguro gana, muy por encima del pre-partido.
    matrix = live_final_matrix(1.3, 1.3, home_goals=2, away_goals=0, minute=80)
    assert _p_home(matrix) > 0.90
