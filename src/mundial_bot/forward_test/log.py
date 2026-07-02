"""Log forward-test de props por jugador en SQLite (data/forward_test.sqlite).

Flujo: `log_prediction` guarda la predicción ANTES del partido (idempotente:
UNIQUE por fixture+player+market → re-loguear no duplica). `settle_fixture`
baja las stats reales post-partido (con el MISMO cache en disco del collector)
y liquida: guarda el valor real y el Brier de las probabilidades. `summary`
resume conteos, Brier medio y MAE de las medias.

Mercados soportados:
  - Conteos (COUNT_STATS del collector: "shots", "sot", "goals", ...): actual =
    valor real del stat; si hay `line` + `pred_prob` se liquida como over
    (actual > line) con Brier.
  - Binarios: "anota" (1+ gol), "anota_o_asiste", "tarjeta" (amarilla o roja):
    actual = 0/1 y Brier sobre `pred_prob`.
  - De EQUIPO (pipeline diario): mercados con prefijo "team_" y player_id=0
    (ej. "team_goals_ou_2.5", "team_1x2_home", "team_se_clasifica_home"). Se
    liquidan con `settle_team_fixture` pasando los totales/resultado reales.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from mundial_bot.collectors.players_wc import (
    COUNT_STATS,
    fetch_fixture_players,
    parse_fixture_players,
)
from mundial_bot.config import DATA_DIR

FORWARD_TEST_DB = DATA_DIR / "forward_test.sqlite"

# Mercados binarios: nombre → cómo se resuelve desde la fila real del jugador.
BINARY_MARKETS = {
    "anota": lambda r: 1.0 if r["goals"] >= 1 else 0.0,
    "anota_o_asiste": lambda r: 1.0 if (r["goals"] + r["assists"]) >= 1 else 0.0,
    "tarjeta": lambda r: 1.0 if (r["yellow"] + r["red"]) >= 1 else 0.0,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS props_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    fixture_id  INTEGER NOT NULL,
    match       TEXT NOT NULL,
    market      TEXT NOT NULL,
    player_id   INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    pred_mean   REAL,
    pred_prob   REAL,
    line        REAL,
    odds        REAL,
    book        TEXT,
    actual      REAL,
    settled_at  TEXT,
    brier       REAL,
    notes       TEXT,
    UNIQUE (fixture_id, player_id, market)
);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Abre (creando si falta) la base del forward test."""
    path = Path(db_path) if db_path is not None else FORWARD_TEST_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log_prediction(
    *,
    fixture_id: int,
    match: str,
    market: str,
    player_id: int,
    player_name: str,
    pred_mean: float | None = None,
    pred_prob: float | None = None,
    line: float | None = None,
    odds: float | None = None,
    book: str | None = None,
    notes: str = "",
    db_path: Path | None = None,
) -> bool:
    """Registra una predicción. Devuelve True si insertó, False si ya existía.

    Idempotente por (fixture_id, player_id, market): re-correr el pipeline no
    duplica ni pisa la predicción original (la primera es la que vale — es un
    forward test, no se reescribe la historia).
    """
    if (market not in BINARY_MARKETS and market not in COUNT_STATS
            and not market.startswith("team_")):
        raise ValueError(f"mercado desconocido: {market!r}")
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO props_log
               (created_at, fixture_id, match, market, player_id, player_name,
                pred_mean, pred_prob, line, odds, book, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), fixture_id, match, market, player_id, player_name,
             pred_mean, pred_prob, line, odds, book, notes),
        )
        return cur.rowcount > 0


def _actual_and_brier(
    row: sqlite3.Row, real: dict[int, dict]
) -> tuple[float, float | None] | None:
    """(actual, brier) de una predicción contra las stats reales; None si no jugó."""
    player = real.get(int(row["player_id"]))
    if player is None:
        return None
    market = str(row["market"])
    if market in BINARY_MARKETS:
        actual = BINARY_MARKETS[market](player)
    else:
        actual = float(player[market])
    brier: float | None = None
    if row["pred_prob"] is not None:
        if market in BINARY_MARKETS:
            outcome = actual
        elif row["line"] is not None:
            outcome = 1.0 if actual > float(row["line"]) else 0.0
        else:
            outcome = None
        if outcome is not None:
            brier = (float(row["pred_prob"]) - outcome) ** 2
    return actual, brier


def settle_fixture(key: str, fixture_id: int, *, db_path: Path | None = None) -> int:
    """Liquida las predicciones pendientes de un fixture. Devuelve filas liquidadas.

    Usa el MISMO cache en disco del collector (si el JSON del fixture ya está
    bajado, cero llamadas a la API). Idempotente: solo toca filas sin liquidar.
    """
    raw = fetch_fixture_players(key, fixture_id)
    real = {
        r["player_id"]: r
        for r in parse_fixture_players(raw, fixture_id=fixture_id)
    }
    settled = 0
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM props_log WHERE fixture_id = ? AND settled_at IS NULL",
            (fixture_id,),
        ).fetchall()
        for row in rows:
            result = _actual_and_brier(row, real)
            if result is None:
                continue  # el jugador no figura en el fixture: queda pendiente
            actual, brier = result
            conn.execute(
                "UPDATE props_log SET actual = ?, brier = ?, settled_at = ? WHERE id = ?",
                (actual, brier, _now(), row["id"]),
            )
            settled += 1
    return settled


def summary(db_path: Path | None = None) -> dict:
    """Resumen del forward test: conteos, Brier medio y MAE de las medias."""
    with _connect(db_path) as conn:
        total, settled = conn.execute(
            "SELECT COUNT(*), COUNT(settled_at) FROM props_log"
        ).fetchone()
        (brier_mean,) = conn.execute(
            "SELECT AVG(brier) FROM props_log WHERE brier IS NOT NULL"
        ).fetchone()
        (mae_mean,) = conn.execute(
            """SELECT AVG(ABS(pred_mean - actual)) FROM props_log
               WHERE pred_mean IS NOT NULL AND actual IS NOT NULL"""
        ).fetchone()
    return {
        "total": int(total),
        "settled": int(settled),
        "pending": int(total - settled),
        "brier_mean": float(brier_mean) if brier_mean is not None else None,
        "mae_mean": float(mae_mean) if mae_mean is not None else None,
    }


# ---------------------------------------------------------------------------
# Mercados de EQUIPO (pipeline diario) — player_id=0, market con prefijo team_
# ---------------------------------------------------------------------------

def _team_actual(
    market: str, line: float | None, actuals: dict
) -> tuple[float, float | None] | None:
    """(valor_real, acierto_0_1) de un mercado de equipo; None si falta el dato.

    ``actuals``: {"goals_total", "corners_total", "yellows_total", "shots_total",
    "sot_total", "result" ('home'/'draw'/'away'), "btts" (0/1),
    "advanced" ('home'/'away' o None si no fue eliminatoria)}.
    """
    body = market[len("team_"):]
    for fam in ("goals", "corners", "yellows", "shots", "sot"):
        if body.startswith(f"{fam}_ou"):
            total = actuals.get(f"{fam}_total")
            if total is None or line is None:
                return None
            return float(total), (1.0 if float(total) > float(line) else 0.0)
    if body.startswith("1x2_"):
        res = actuals.get("result")
        if res is None:
            return None
        hit = 1.0 if res == body.rsplit("_", 1)[1] else 0.0
        return hit, hit
    if body == "btts":
        v = actuals.get("btts")
        return (float(v), float(v)) if v is not None else None
    if body.startswith("se_clasifica_"):
        adv = actuals.get("advanced")
        if adv is None:
            return None
        hit = 1.0 if adv == body.rsplit("_", 1)[1] else 0.0
        return hit, hit
    return None


def settle_team_fixture(
    fixture_id: int, actuals: dict, *, db_path: Path | None = None
) -> int:
    """Liquida los mercados de EQUIPO pendientes de un fixture (idempotente).

    El caller (pipeline post-día) arma ``actuals`` desde el fixture real cacheado.
    Brier solo si la fila tenía pred_prob. Devuelve filas liquidadas.
    """
    settled = 0
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM props_log
               WHERE fixture_id=? AND player_id=0 AND settled_at IS NULL""",
            (fixture_id,),
        ).fetchall()
        for row in rows:
            out = _team_actual(str(row["market"]), row["line"], actuals)
            if out is None:
                continue
            actual, hit = out
            brier = None
            if row["pred_prob"] is not None and hit is not None:
                brier = (float(row["pred_prob"]) - hit) ** 2
            conn.execute(
                "UPDATE props_log SET actual=?, brier=?, settled_at=? WHERE id=?",
                (actual, brier, _now(), row["id"]),
            )
            settled += 1
    return settled
