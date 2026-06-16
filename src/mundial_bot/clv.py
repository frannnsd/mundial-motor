"""CLV (Closing Line Value): mide si el bot es SHARP.

Cuando el bot marca un pick, guarda la cuota de APERTURA. Cerca del inicio del partido
captura la cuota de CIERRE. Si el cierre es MÁS BAJO que lo que marcamos (la cuota se
acortó hacia nuestro pick), el mercado nos dio la razón → CLV positivo. Ganarle al
cierre de forma consistente es la prueba de fuego de un modelo que tiene edge —
independientemente de si el partido sale o no.

CLV% = cuota_apertura / cuota_cierre − 1   (positivo = le ganamos al cierre).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mundial_bot.config import DATA_DIR

CLV_DB = DATA_DIR / "clv.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clv (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at   TEXT NOT NULL,
    fixture_id  INTEGER NOT NULL,
    match       TEXT NOT NULL,
    market      TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    pick        TEXT NOT NULL,
    open_odds   REAL NOT NULL,
    open_book   TEXT NOT NULL,
    closed_at   TEXT,
    close_odds  REAL,
    close_book  TEXT,
    clv_pct     REAL,
    UNIQUE(fixture_id, market, outcome)
);
"""


@dataclass
class ClvSummary:
    n_tracked: int          # picks con cuota de apertura
    n_closed: int           # picks con cuota de cierre (muestra de CLV)
    positive: int           # cuántos le ganaron al cierre
    avg_clv: float | None   # CLV promedio


class ClvStore:
    def __init__(self, db_path: str | Path = CLV_DB):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def open_pick(self, *, opened_at: str, fixture_id: int, match: str, market: str,
                  outcome: str, pick: str, open_odds: float, open_book: str) -> int:
        """Registra la cuota de APERTURA de un pick. Idempotente (no pisa la apertura)."""
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO clv
               (opened_at, fixture_id, match, market, outcome, pick, open_odds, open_book)
               VALUES (?,?,?,?,?,?,?,?)""",
            (opened_at, fixture_id, match, market, outcome, pick, open_odds, open_book),
        )
        self.conn.commit()
        return cur.rowcount

    def open_for_fixture(self, fixture_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM clv WHERE fixture_id=?", (fixture_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def set_close(self, row_id: int, *, closed_at: str, close_odds: float,
                  close_book: str, open_odds: float) -> None:
        """Captura/actualiza la cuota de CIERRE (la última antes del inicio = el cierre)."""
        clv = open_odds / close_odds - 1.0 if close_odds > 0 else 0.0
        self.conn.execute(
            "UPDATE clv SET closed_at=?, close_odds=?, close_book=?, clv_pct=? WHERE id=?",
            (closed_at, close_odds, close_book, clv, row_id),
        )
        self.conn.commit()

    def summary(self) -> ClvSummary:
        rows = self.conn.execute("SELECT clv_pct FROM clv").fetchall()
        closed = [r["clv_pct"] for r in rows if r["clv_pct"] is not None]
        positive = sum(1 for c in closed if c > 0)
        avg = sum(closed) / len(closed) if closed else None
        return ClvSummary(
            n_tracked=len(rows), n_closed=len(closed), positive=positive, avg_clv=avg
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> ClvStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def format_clv(summary: ClvSummary) -> str:
    """Reporte de CLV para Telegram (HTML)."""
    if summary.n_closed == 0:
        return (
            "📈 <b>CLV (¿el bot le gana al mercado?)</b>\n\n"
            f"Siguiendo {summary.n_tracked} picks, todavía sin cierres para comparar. "
            "Se mide cuando los partidos marcados arrancan. ⏳"
        )
    pct_pos = summary.positive / summary.n_closed
    avg = summary.avg_clv or 0.0
    verdict = (
        "🟢 le está ganando al cierre (señal de que tiene edge)" if avg > 0.005
        else "🟡 a la par del mercado" if avg > -0.005
        else "🔴 el mercado le gana (cuidado)"
    )
    return (
        "📈 <b>CLV — ¿el bot le gana al mercado?</b>\n\n"
        f"Picks con cierre: <b>{summary.n_closed}</b>\n"
        f"Le ganaron al cierre: <b>{summary.positive}/{summary.n_closed}</b> "
        f"({pct_pos:.0%})\n"
        f"CLV promedio: <b>{avg:+.1%}</b>\n\n{verdict}\n"
        "<i>CLV = cuánto mejor fue tu cuota vs la de cierre. Positivo sostenido = sharp.</i>"
    )
