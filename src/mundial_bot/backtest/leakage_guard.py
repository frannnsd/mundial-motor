"""Guard anti-leakage point-in-time.

Regla sagrada (CLAUDE.md / guardrail #2): al scorear un partido, NINGUNA feature puede
usar datos con timestamp >= kickoff. Este módulo convierte esa regla en una ASERCIÓN:
si un DataFrame de features contiene una fila con fecha >= kickoff, `assert_point_in_time`
levanta `LeakageError` y hace fallar el test.

Uso típico en un backtest walk-forward:

    train = df[df["date"] < kickoff]
    assert_point_in_time(train, kickoff, label="córners")   # red de seguridad
    model = CornersModel.from_events(train)                 # (o from_events(df, as_of=kickoff))
"""

from __future__ import annotations

import pandas as pd


class LeakageError(AssertionError):
    """Se usó un dato con timestamp >= kickoff para predecir ese mismo partido."""


def max_feature_date(
    events: pd.DataFrame | None, *, date_col: str = "date"
) -> pd.Timestamp | None:
    """Fecha más reciente presente en los datos de features (None si no hay fechas)."""
    if events is None or len(events) == 0 or date_col not in getattr(events, "columns", []):
        return None
    dates = pd.to_datetime(events[date_col], errors="coerce")
    return dates.max() if dates.notna().any() else None


def assert_point_in_time(
    events: pd.DataFrame | None,
    as_of: pd.Timestamp | str | None,
    *,
    date_col: str = "date",
    label: str = "features",
) -> None:
    """Levanta LeakageError si `events` contiene datos con fecha >= `as_of` (kickoff).

    Con `as_of=None` no hace nada (no hay kickoff contra el cual chequear).
    """
    if as_of is None:
        return
    as_of_ts = pd.Timestamp(as_of)
    mx = max_feature_date(events, date_col=date_col)
    if mx is not None and mx >= as_of_ts:
        raise LeakageError(
            f"LEAKAGE en '{label}': hay datos con fecha {mx} >= kickoff {as_of_ts}. "
            "Toda feature debe usar solo partidos ANTERIORES al kickoff."
        )
