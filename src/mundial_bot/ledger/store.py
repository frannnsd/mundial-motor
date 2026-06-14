"""Ledger de apuestas en SQLite — Agente 6.

Registra cada pick sugerido y, al cerrarse el partido, su resultado. De ahí salen
las métricas que importan de verdad:
  - **ROI / yield**: ganancia / total apostado.
  - **CLV (Closing Line Value)**: ¿conseguimos mejor cuota que la de cierre del
    mercado? Es el mejor predictor de rentabilidad a largo plazo. Si el CLV es
    negativo pero ganamos plata, es suerte y va a revertir.

Sin dependencias externas: usa el módulo `sqlite3` de la stdlib.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mundial_bot.config import DATA_DIR

DEFAULT_DB = DATA_DIR / "ledger.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    match        TEXT NOT NULL,
    market       TEXT NOT NULL,
    selection    TEXT NOT NULL,
    odds         REAL NOT NULL,
    bookmaker    TEXT,
    model_prob   REAL NOT NULL,
    fair_prob    REAL,
    edge         REAL NOT NULL,
    stake        REAL NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'single',
    status       TEXT NOT NULL DEFAULT 'pending',
    closing_odds REAL,
    pnl          REAL
);
"""


@dataclass
class LedgerSummary:
    total: int
    settled: int
    won: int
    staked: float
    returned: float
    roi: float          # (returned - staked) / staked
    clv_avg: float | None  # promedio de CLV de picks con closing_odds


class Ledger:
    """Almacén de picks y resultados."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def record(
        self,
        *,
        created_at: str,
        match: str,
        market: str,
        selection: str,
        odds: float,
        model_prob: float,
        edge: float,
        stake: float,
        bookmaker: str = "?",
        fair_prob: float | None = None,
        kind: str = "single",
    ) -> int:
        """Registra un pick (pending). Devuelve su id."""
        cur = self.conn.execute(
            """INSERT INTO picks
               (created_at, match, market, selection, odds, bookmaker,
                model_prob, fair_prob, edge, stake, kind)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (created_at, match, market, selection, odds, bookmaker,
             model_prob, fair_prob, edge, stake, kind),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def settle(self, pick_id: int, *, won: bool, closing_odds: float | None = None) -> None:
        """Cierra un pick con su resultado y calcula el PnL."""
        row = self.conn.execute("SELECT odds, stake FROM picks WHERE id=?", (pick_id,)).fetchone()
        if row is None:
            raise KeyError(f"No existe el pick {pick_id}")
        pnl = row["stake"] * (row["odds"] - 1.0) if won else -row["stake"]
        self.conn.execute(
            "UPDATE picks SET status=?, closing_odds=?, pnl=? WHERE id=?",
            ("won" if won else "lost", closing_odds, pnl, pick_id),
        )
        self.conn.commit()

    def open_picks(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM picks WHERE status='pending'").fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> LedgerSummary:
        rows = self.conn.execute("SELECT * FROM picks").fetchall()
        settled = [r for r in rows if r["status"] in ("won", "lost")]
        staked = sum(r["stake"] for r in settled)
        returned = sum(
            r["stake"] * r["odds"] for r in settled if r["status"] == "won"
        )
        roi = (returned - staked) / staked if staked > 0 else 0.0

        clv_rows = [r for r in rows if r["closing_odds"]]
        clv_avg = (
            sum(r["odds"] / r["closing_odds"] - 1.0 for r in clv_rows) / len(clv_rows)
            if clv_rows
            else None
        )
        return LedgerSummary(
            total=len(rows),
            settled=len(settled),
            won=sum(1 for r in settled if r["status"] == "won"),
            staked=round(staked, 2),
            returned=round(returned, 2),
            roi=roi,
            clv_avg=clv_avg,
        )

    def close(self) -> None:
        self.conn.close()
