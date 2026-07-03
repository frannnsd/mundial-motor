"""Motor MLB unificado para el pipeline vivo (M4) — espejo béisbol de wc/engine.py.

Construye el estado de los cerebros MLB caminando TODO el histórico (2015→hoy,
en orden cronológico, revelando cada día — point-in-time hacia adelante, SIN
puntuar: solo estado, ~30-60 s) y predice juegos FUTUROS con la mixtura
unificada (pesos de data/mlb_weights.json, generados por la validación).

El histórico sale de collectors.mlb_data cache-primero; si en la nube no hay
cache (disco efímero de Render), build_mlb_table reconstruye con fetch_season
por año (12 llamadas, gratis, quedan cacheadas en el disco efímero).

El estado se reconstruye en cada corrida: sin estado persistente que pueda
desincronizarse (misma filosofía que WcEngine).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mundial_bot.collectors.mlb_data import build_mlb_table
from mundial_bot.config import DATA_DIR
from mundial_bot.research.distributions import count_pmf
from mundial_bot.research.mlb import GRID_MLB, QUANTITIES_MLB, MlbConfig, MlbState

logger = logging.getLogger(__name__)

MLB_WEIGHTS_FILE = DATA_DIR / "mlb_weights.json"
_UNIFORM_BRAINS = ("A", "B", "C")


def load_mlb_weights(path: Path = MLB_WEIGHTS_FILE) -> dict[str, dict[str, float]]:
    """Pesos del unificado MLB ({qty: {cerebro: peso}}, los genera el orquestador).

    Fallback honesto: si el archivo no existe todavía, pesos uniformes A/B/C
    con warning (el bobo queda afuera: es la vara, no un cerebro para producción).
    """
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    logger.warning("Sin %s: uso pesos uniformes A/B/C (el orquestador los genera "
                   "con la validación).", path.name)
    w = 1.0 / len(_UNIFORM_BRAINS)
    return {q: {b: w for b in _UNIFORM_BRAINS} for q in QUANTITIES_MLB}


class MlbLiveEngine:
    """Estado entrenado sobre el histórico MLB + predicción unificada de juegos futuros."""

    def __init__(
        self,
        table: pd.DataFrame | None = None,
        *,
        config: MlbConfig | None = None,
        weights: dict[str, dict[str, float]] | None = None,
    ):
        self.cfg = config or MlbConfig()
        self.weights = weights or load_mlb_weights()
        self.state = MlbState(self.cfg)
        df = table if table is not None else build_mlb_table()
        if df.empty:
            raise RuntimeError("Tabla MLB vacía: ni cache local ni Stats API disponibles.")
        self.history = df.sort_values("date").reset_index(drop=True)
        self._walk(self.history)

    def _walk(self, df: pd.DataFrame) -> None:
        """Revela todo el histórico en orden (solo estado; acá no se predice nada)."""
        n = len(df)
        i = 0
        while i < n:
            day = df["date"].iloc[i]
            j = i
            while j < n and df["date"].iloc[j] == day:
                j += 1
            for k in range(i, j):
                self.state.reveal(df.iloc[k], day)
            self.state.end_day(day)
            i = j
        self.last_date = df["date"].max() if n else None

    def predict_game(
        self,
        home: str,
        away: str,
        venue: str,
        starter_h_id: int | None,
        starter_a_id: int | None,
        when: pd.Timestamp | str,
    ) -> dict:
        """Predicción unificada de un juego FUTURO: {"pmfs": {qty: pmf}, "means": ...}.

        La row sintética NO lleva columnas de resultado: MlbState.predict solo
        encola features del GLM cuando hay actual, así que un juego futuro no
        ensucia el fit; se limpia _pending igual, como cinturón y tiradores.
        """
        when = pd.Timestamp(when)
        day = pd.Timestamp(when.date())
        row = pd.Series({
            "home_team": home, "away_team": away, "venue": venue or "",
            "season": str(when.year),
            "starter_h_id": starter_h_id, "starter_a_id": starter_a_id,
        })
        preds = self.state.predict(row, day)
        self.state._pending.clear()  # noqa: SLF001 — reuso deliberado (ver docstring)

        pmfs: dict[str, np.ndarray] = {}
        means: dict[str, float] = {}
        for q in QUANTITIES_MLB:
            family = q.rsplit("_", 1)[0]  # "runs_f5_h" → "runs_f5"
            k_max = GRID_MLB[family]
            w = self.weights.get(q) or {b: 1.0 / len(_UNIFORM_BRAINS)
                                        for b in _UNIFORM_BRAINS}
            total_w = sum(w.values()) or 1.0
            mix = np.zeros(k_max + 1)
            mean_mix = 0.0
            for brain, wb in w.items():
                m, v = preds[brain][q]
                mix = mix + count_pmf(m, v, k_max) * (wb / total_w)
                mean_mix += (wb / total_w) * m
            pmfs[q] = mix
            means[q] = mean_mix
        return {"pmfs": pmfs, "means": means, "weights": self.weights}
