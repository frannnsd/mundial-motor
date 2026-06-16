"""Modo EN VIVO: ajusta las probabilidades al marcador y minuto actual.

Del xG pre-partido del Dixon-Coles, los goles que FALTAN se escalan por el tiempo
restante (minutos que quedan / 90). El marcador FINAL = marcador actual + goles
restantes (Poisson). Con esa matriz de marcador final se derivan TODOS los mercados de
goles desde el estado actual (1X2, totales, hándicaps, ambos marcan, etc.).

Aproximación honesta: no modela el "game state" (un equipo que va perdiendo suele
empujar más). Es un ajuste de tiempo+marcador, no una simulación táctica. Córners y
tarjetas no se ajustan en vivo (haría falta el conteo en vivo, que no tenemos).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from mundial_bot.models.goals_model import GoalsModel, GoalsModelError
from mundial_bot.models.market_book import MarketBook, _goals_selections, real_odd

FULL_MATCH_MIN = 90.0
_MAX_EXTRA_GOALS = 10


def live_final_matrix(
    home_xg: float, away_xg: float, home_goals: int, away_goals: int, minute: float
) -> np.ndarray:
    """Matriz P[i,j] del marcador FINAL dado el estado actual. Filas=local, col=visita."""
    remaining = max(0.0, (FULL_MATCH_MIN - min(minute, FULL_MATCH_MIN)) / FULL_MATCH_MIN)
    lam_h = max(1e-9, home_xg * remaining)
    lam_a = max(1e-9, away_xg * remaining)
    n = max(home_goals, away_goals) + _MAX_EXTRA_GOALS + 1
    rem_h = poisson.pmf(np.arange(n), lam_h)   # dist de goles que faltan (índice = restantes)
    rem_a = poisson.pmf(np.arange(n), lam_a)
    matrix = np.zeros((n, n))
    for ri in range(n - home_goals):
        for rj in range(n - away_goals):
            matrix[home_goals + ri, away_goals + rj] = rem_h[ri] * rem_a[rj]
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def build_live_book(
    home: str, away: str, *, goals: GoalsModel,
    home_goals: int, away_goals: int, minute: float, neutral: bool = True,
    match_name: str | None = None,
) -> MarketBook:
    """Libro de mercados EN VIVO (mercados de goles ajustados al marcador+minuto)."""
    _, home_xg, away_xg = goals.score_matrix(home, away, neutral=neutral)
    matrix = live_final_matrix(home_xg, away_xg, home_goals, away_goals, minute)
    remaining = max(0.0, (FULL_MATCH_MIN - min(minute, FULL_MATCH_MIN)) / FULL_MATCH_MIN)
    rem_home_xg, rem_away_xg = home_xg * remaining, away_xg * remaining
    selections = _goals_selections(matrix, home, away, rem_home_xg, rem_away_xg)

    n = matrix.shape[0]
    m = np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1)
    p_home = float(matrix[m > 0].sum())
    p_draw = float(matrix[m == 0].sum())
    p_away = float(matrix[m < 0].sum())
    return MarketBook(
        match=match_name or f"{home} vs {away}",
        home=home, away=away, home_xg=rem_home_xg, away_xg=rem_away_xg,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        exp_goals=home_goals + away_goals + rem_home_xg + rem_away_xg,
        elo_home=p_home, elo_draw=p_draw, elo_away=p_away,
        dc_home=p_home, dc_draw=p_draw, dc_away=p_away,
        selections=selections,
    )


def format_live_book(
    book: MarketBook, *, home_goals: int, away_goals: int, minute: float,
    odds: dict | None = None,
) -> str:
    """Texto del panorama EN VIVO: estado + mercados del marcador final desde ahora."""
    rem = int(max(0, FULL_MATCH_MIN - min(minute, FULL_MATCH_MIN)))
    head = (
        f"🔴 EN VIVO — {book.match}\n"
        f"Marcador: {book.home} {home_goals}-{away_goals} {book.away} · minuto ~{int(minute)} "
        f"(quedan ~{rem}')\n"
        f"Resultado FINAL desde ahora: {book.home} {book.p_home:.0%} / "
        f"X {book.p_draw:.0%} / {book.away} {book.p_away:.0%}\n"
        f"Goles que faltan (estimado): {book.home_xg + book.away_xg:.2f} "
        f"→ total esperado ~{book.exp_goals:.1f}\n"
        "Ajuste de tiempo+marcador (no modela presión por ir perdiendo). Córners/"
        "tarjetas no se ajustan en vivo.\n"
    )
    lines = [head]
    for market, sels in book.by_market().items():
        lines.append(f"\n[{market}]")
        for s in sels:
            extra = f" · push {s.push:.0%}" if s.push > 0.01 else ""
            found = real_odd(s, odds)
            casa = f" · casa {found[0]:.2f} ({found[1]})" if found else ""
            lines.append(
                f"  {s.pick}: {s.prob:.0%} (cuota s/modelo {s.fair:.2f}){casa}{extra}"
            )
    return "\n".join(lines)


def live_analysis(
    goals: GoalsModel, home: str, away: str, *,
    home_goals: int, away_goals: int, minute: float, match_name: str | None = None,
) -> str:
    """Análisis en vivo listo para Telegram (o mensaje claro si no se puede predecir)."""
    if not goals.can_predict(home, away):
        return f"(NO tengo a {home} o {away} en el modelo para analizar en vivo.)"
    try:
        book = build_live_book(
            home, away, goals=goals, home_goals=home_goals, away_goals=away_goals,
            minute=minute, match_name=match_name,
        )
    except GoalsModelError as exc:
        return f"(No pude armar el en vivo: {exc})"
    return format_live_book(book, home_goals=home_goals, away_goals=away_goals, minute=minute)
