"""Registro de las apuestas REALES de Franco — disciplina + ROI de verdad.

Distinto del log de predicciones (lo que el bot piensa): acá Franco anota lo que
APOSTÓ de verdad (monto + cuota), y al cerrarse sabe su ROI real. Lo que separa a
los que ganan de los que se funden: registrar todo, stake fijo, sin chasing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mundial_bot.config import DATA_DIR

BETS_DB = DATA_DIR / "bets.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    description TEXT NOT NULL,
    stake       REAL NOT NULL,
    odds        REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    pnl         REAL
);
"""


@dataclass
class BetSummary:
    total: int
    settled: int
    won: int
    staked: float
    returned: float

    @property
    def roi(self) -> float:
        return (self.returned - self.staked) / self.staked if self.staked > 0 else 0.0

    @property
    def profit(self) -> float:
        return self.returned - self.staked


class BetStore:
    def __init__(self, db_path: str | Path = BETS_DB):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def log(self, *, created_at: str, description: str, stake: float, odds: float) -> int:
        if stake <= 0 or odds <= 1.0:
            raise ValueError("Stake debe ser > 0 y cuota > 1.0")
        cur = self.conn.execute(
            "INSERT INTO bets (created_at, description, stake, odds) VALUES (?,?,?,?)",
            (created_at, description, stake, odds),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def settle(self, bet_id: int, *, won: bool) -> None:
        row = self.conn.execute("SELECT stake, odds FROM bets WHERE id=?", (bet_id,)).fetchone()
        if row is None:
            raise KeyError(f"No existe la apuesta {bet_id}")
        pnl = row["stake"] * (row["odds"] - 1.0) if won else -row["stake"]
        self.conn.execute(
            "UPDATE bets SET status=?, pnl=? WHERE id=?",
            ("won" if won else "lost", pnl, bet_id),
        )
        self.conn.commit()

    def open_bets(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM bets WHERE status='pending' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> BetSummary:
        rows = self.conn.execute("SELECT * FROM bets").fetchall()
        settled = [r for r in rows if r["status"] in ("won", "lost")]
        staked = sum(r["stake"] for r in settled)
        returned = sum(r["stake"] * r["odds"] for r in settled if r["status"] == "won")
        return BetSummary(
            total=len(rows), settled=len(settled),
            won=sum(1 for r in settled if r["status"] == "won"),
            staked=round(staked, 2), returned=round(returned, 2),
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> BetStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_bet_command(text: str) -> tuple[float, float, str]:
    """Parsea '/apuesta 5 2.10 Argentina gana' → (stake, odds, descripción)."""
    parts = text.split()
    # parts[0] = '/apuesta'
    if len(parts) < 4:
        raise ValueError("Formato: /apuesta <monto> <cuota> <descripción>")
    stake = float(parts[1])
    odds = float(parts[2])
    description = " ".join(parts[3:])
    return stake, odds, description


def format_roi(summary: BetSummary) -> str:
    if summary.settled == 0:
        return (
            "💰 <b>TU ROI</b>\n\nTodavía no cerraste ninguna apuesta.\n"
            "Anotá con: <code>/apuesta 5 2.10 Argentina gana</code>"
        )
    sign = "🟢" if summary.profit >= 0 else "🔴"
    return (
        "💰 <b>TU ROI REAL</b>\n"
        f"{sign} Ganancia: <b>${summary.profit:+.2f}</b> · ROI <b>{summary.roi:+.1%}</b>\n"
        f"Apuestas: {summary.won}/{summary.settled} ganadas · "
        f"apostado ${summary.staked:.2f} · cobrado ${summary.returned:.2f}"
    )
