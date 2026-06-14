"""Autoevaluación: guarda predicciones, baja resultados reales, califica y reporta.

El loop:
  1. Al predecir un partido, se loguea cada mercado (ganador/goles/córners/tarjetas/BTTS).
  2. Cuando el partido termina, se baja el resultado real (marcador + estadísticas).
  3. Cada predicción se marca acierto/error.
  4. Se calcula el % de acierto por mercado + Brier (calibración) → balance por Telegram.

Esos aciertos/errores después alimentan la calibración (ver brain/calibración).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import requests

from mundial_bot.collectors.team_stats import _extract
from mundial_bot.config import DATA_DIR
from mundial_bot.report import MatchReport

PREDICTIONS_DB = DATA_DIR / "predictions.sqlite"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
TIMEOUT_S = 25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pred_date   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    fixture_id  INTEGER NOT NULL,
    match       TEXT NOT NULL,
    market      TEXT NOT NULL,
    side        TEXT NOT NULL,
    line        REAL,
    pick        TEXT NOT NULL,
    prob        REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    actual      REAL,
    UNIQUE(fixture_id, market, pred_date)
);
"""

_MARKETS = ("winner", "goals", "btts", "corners", "cards")


@dataclass
class Balance:
    n: int
    correct: int
    by_market: dict[str, tuple[int, int]]  # market -> (aciertos, total)
    brier: float | None

    @property
    def hit_rate(self) -> float:
        return self.correct / self.n if self.n else 0.0


class PredictionStore:
    def __init__(self, db_path: str | Path = PREDICTIONS_DB):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def log_report(self, fixture_id: int | None, report: MatchReport, *, pred_date: str,
                   created_at: str) -> int:
        """Loguea cada mercado de un reporte. Idempotente por (fixture, mercado, fecha)."""
        if fixture_id is None:
            return 0
        picks = {
            "winner": report.winner, "goals": report.goals, "btts": report.btts,
            "corners": report.corners, "cards": report.cards,
        }
        n = 0
        for market, pick in picks.items():
            if pick is None:
                continue
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO predictions
                   (pred_date, created_at, fixture_id, match, market, side, line, pick, prob)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (pred_date, created_at, fixture_id, report.match, market,
                 pick.side, pick.line, pick.pick, pick.prob),
            )
            n += cur.rowcount
        self.conn.commit()
        return n

    def pending_fixture_ids(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT fixture_id FROM predictions WHERE status='pending'"
        ).fetchall()
        return [r["fixture_id"] for r in rows]

    def pending_for_fixture(self, fixture_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM predictions WHERE status='pending' AND fixture_id=?", (fixture_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def settle(self, pred_id: int, *, status: str, actual: float | None) -> None:
        self.conn.execute(
            "UPDATE predictions SET status=?, actual=? WHERE id=?", (status, actual, pred_id)
        )
        self.conn.commit()

    def balance(self) -> Balance:
        rows = self.conn.execute(
            "SELECT market, prob, status FROM predictions WHERE status IN ('correct','wrong')"
        ).fetchall()
        by_market: dict[str, list[int]] = {m: [0, 0] for m in _MARKETS}
        brier_sum = 0.0
        correct = 0
        for r in rows:
            hit = 1 if r["status"] == "correct" else 0
            by_market.setdefault(r["market"], [0, 0])
            by_market[r["market"]][0] += hit
            by_market[r["market"]][1] += 1
            correct += hit
            brier_sum += (r["prob"] - hit) ** 2
        n = len(rows)
        return Balance(
            n=n, correct=correct,
            by_market={m: (a, t) for m, (a, t) in by_market.items() if t > 0},
            brier=(brier_sum / n) if n else None,
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> PredictionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def grade_outcome(market: str, side: str, line: float | None, result: dict) -> tuple[str, float]:
    """Devuelve (status, actual) para una predicción dado el resultado real."""
    hs, as_ = result["home_score"], result["away_score"]
    if market == "winner":
        outcome = "home" if hs > as_ else ("away" if as_ > hs else "draw")
        return ("correct" if side == outcome else "wrong"), float(hs > as_) - float(as_ > hs)
    if market == "goals":
        total = hs + as_
        over = total > (line or 0)
        return ("correct" if (side == "over") == over else "wrong"), float(total)
    if market == "btts":
        both = hs > 0 and as_ > 0
        return ("correct" if (side == "yes") == both else "wrong"), float(both)
    if market == "corners":
        total = result.get("corners", 0)
        over = total > (line or 0)
        return ("correct" if (side == "over") == over else "wrong"), float(total)
    if market == "cards":
        total = result.get("cards", 0)
        over = total > (line or 0)
        return ("correct" if (side == "over") == over else "wrong"), float(total)
    return "void", 0.0


def fetch_result(key: str, fixture_id: int) -> dict | None:
    """Baja el resultado final de un partido. None si todavía no terminó."""
    fx = requests.get(
        f"{API_FOOTBALL_BASE}/fixtures", headers={"x-apisports-key": key},
        params={"id": fixture_id}, timeout=TIMEOUT_S,
    ).json()
    resp = fx.get("response", [])
    if not resp:
        return None
    item = resp[0]
    if (item.get("fixture", {}).get("status", {}) or {}).get("short") != "FT":
        return None  # no terminó
    goals = item.get("goals", {})
    home_score, away_score = goals.get("home"), goals.get("away")
    if home_score is None or away_score is None:
        return None

    stats = requests.get(
        f"{API_FOOTBALL_BASE}/fixtures/statistics", headers={"x-apisports-key": key},
        params={"fixture": fixture_id}, timeout=TIMEOUT_S,
    ).json().get("response", [])
    corners = cards = 0
    for entry in stats:
        ex = _extract(entry.get("statistics", []))
        corners += ex["corners"]
        cards += ex["cards"]
    return {
        "home_score": int(home_score), "away_score": int(away_score),
        "corners": corners, "cards": cards,
    }


def grade_pending(key: str, *, store: PredictionStore | None = None) -> int:
    """Califica todas las predicciones pendientes cuyo partido ya terminó. Devuelve cuántas."""
    own = store is None
    store = store or PredictionStore()
    graded = 0
    try:
        for fixture_id in store.pending_fixture_ids():
            result = fetch_result(key, fixture_id)
            if result is None:
                continue
            for pred in store.pending_for_fixture(fixture_id):
                status, actual = grade_outcome(pred["market"], pred["side"], pred["line"], result)
                store.settle(pred["id"], status=status, actual=actual)
                graded += 1
    finally:
        if own:
            store.close()
    return graded


def format_balance(balance: Balance) -> str:
    """Arma el mensaje de balance para Telegram (HTML)."""
    if balance.n == 0:
        return "📊 <b>BALANCE</b>\n\nTodavía no hay partidos calificados. Esperá a que jueguen. ⏳"
    labels = {
        "winner": "🏆 Ganador", "goals": "⚽ Goles", "btts": "🤝 Ambos marcan",
        "corners": "🚩 Córners", "cards": "🟨 Tarjetas",
    }
    lines = [
        "📊 <b>BALANCE DEL BOT</b>",
        f"✅ Aciertos: <b>{balance.correct}/{balance.n}</b> ({balance.hit_rate:.0%})",
        "",
    ]
    for market, label in labels.items():
        if market in balance.by_market:
            hit, total = balance.by_market[market]
            lines.append(f"{label}: {hit}/{total} ({hit / total:.0%})")
    if balance.brier is not None:
        lines.append(f"\n🎯 Calibración (Brier): {balance.brier:.3f}  (más bajo = mejor)")
    return "\n".join(lines)
