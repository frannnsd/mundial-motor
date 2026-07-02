"""Motor unificado de SELECCIONES para el pipeline vivo del Mundial.

Construye el estado de los cerebros caminando TODO el histórico internacional
(en orden cronológico, revelando cada día — point-in-time hacia adelante) y
predice partidos FUTUROS con la mixtura unificada (pesos de wc_validation).

El estado se reconstruye desde la tabla cacheada en cada corrida (~segundos):
sin estado persistente que pueda desincronizarse. Tras cada jornada, la corrida
post-partido re-baja la tabla y el próximo build ya la incluye (Parte 5.4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mundial_bot.config import DATA_DIR
from mundial_bot.research.brains import FAMILIES, BrainConfig, LeagueState, col_key
from mundial_bot.research.distributions import count_pmf, quantity_grid
from mundial_bot.research.wc_validation import NT_BASE_CONFIG

WEIGHTS_FILE = DATA_DIR / "nt_weights.json"
WC_HOSTS = {"USA", "Mexico", "Canada"}


class WcEngine:
    """Estado entrenado sobre selecciones + predicción unificada de partidos futuros."""

    def __init__(
        self,
        nt_df: pd.DataFrame,
        *,
        config: BrainConfig | None = None,
        weights: dict[str, dict[str, float]] | None = None,
    ):
        self.cfg = config or NT_BASE_CONFIG
        self.weights = weights or load_nt_weights()
        self.state = LeagueState(self.cfg)
        self._walk(nt_df)

    def _walk(self, df: pd.DataFrame) -> None:
        """Revela todo el histórico en orden (solo estado; acá no se predice nada)."""
        df = df.sort_values("date").reset_index(drop=True)
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

    def predict_match(
        self, home: str, away: str, *, when: pd.Timestamp, neutral: bool | None = None
    ) -> dict:
        """Predicción unificada de un partido FUTURO del Mundial.

        Devuelve {"pmfs": {cantidad: pmf}, "means": {cantidad: media}, "weights": ...}.
        ``neutral``: por defecto, True salvo que el local sea anfitrión (USA/Mex/Can).
        """
        if neutral is None:
            neutral = home not in WC_HOSTS
        row = pd.Series({
            "home_team": home, "away_team": away,
            "season": str(when.year), "match_type": "mundial", "neutral": neutral,
        })
        preds = self.state.predict(row, pd.Timestamp(when.date()))

        pmfs: dict[str, np.ndarray] = {}
        means: dict[str, float] = {}
        for family in FAMILIES:
            k_max = quantity_grid(family)
            for side in ("h", "a"):
                q = col_key(family, side)
                w = self.weights.get(q, {"bobo": 1.0})
                mix = None
                mean_mix = 0.0
                for brain, wb in w.items():
                    m, v = preds[brain][q]
                    pmf = count_pmf(m, v, k_max)
                    mix = pmf * wb if mix is None else mix + pmf * wb
                    mean_mix += wb * m
                pmfs[q] = mix
                means[q] = mean_mix
        return {"pmfs": pmfs, "means": means, "weights": self.weights, "neutral": neutral}


def load_nt_weights(path: Path = WEIGHTS_FILE) -> dict[str, dict[str, float]]:
    """Pesos NT persistidos por la validación (fallback: bobo puro con warning)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {col_key(f, s): {"bobo": 1.0} for f in FAMILIES for s in ("h", "a")}


def save_nt_weights(weights: dict, path: Path = WEIGHTS_FILE) -> None:
    path.write_text(json.dumps(weights, indent=1, ensure_ascii=False), encoding="utf-8")
