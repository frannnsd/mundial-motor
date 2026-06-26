"""Entrena el cerebro UNA vez y lo guarda en data/brain.pkl.

En la nube (Render), esto corre en el build: así el servicio arranca cargando el
pickle (~0s, poca memoria) en vez de entrenar 25s en cada arranque — clave para
entrar en el plan gratis. Localmente no hace falta (si no existe el pickle, la API
entrena al vuelo como siempre).
"""

from __future__ import annotations

import json
import pickle

from mundial_bot.brain import load_brain
from mundial_bot.config import DATA_DIR


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Entrenando el cerebro (Elo + Dixon-Coles)...", flush=True)
    brain = load_brain()

    # Guard: el simulador NO sirve sin modelo de goles. Si no entrenó, fallar fuerte
    # acá (en vez de shippear un cerebro mudo que rompe /simulate en producción).
    goals = getattr(getattr(brain, "models", None), "goals", None)
    n_teams = len(getattr(goals, "teams", ()) or ()) if goals is not None else 0
    if goals is None or n_teams <= 50:
        raise SystemExit(
            f"ABORT: el modelo de goles no entrenó (goals={goals!r}, equipos={n_teams}). "
            "No congelo un cerebro sin goles."
        )
    print(f"Modelo de goles OK: {n_teams} equipos.", flush=True)

    path = DATA_DIR / "brain.pkl"
    with path.open("wb") as f:
        pickle.dump(brain, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"OK -> {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Backtest pre-computado: así la nube no re-entrena en runtime (memoria + velocidad).
    print("Pre-calculando el backtest...", flush=True)
    from mundial_bot.backtest.sim_backtest import run_backtest

    bt = run_backtest()
    (DATA_DIR / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    print(f"backtest.json -> {bt.get('n', 0)} partidos", flush=True)


if __name__ == "__main__":
    main()
