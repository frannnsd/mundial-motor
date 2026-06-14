"""Tests del notificador: formateo + envío dry-run (Agente 5)."""

from __future__ import annotations

from mundial_bot.notify.formatting import (
    format_daily_card,
    format_single,
    render_selection,
)
from mundial_bot.notify.telegram_bot import _split_message, send_telegram
from mundial_bot.staking.kelly import StakedPick
from mundial_bot.staking.parlays import build_parlay
from mundial_bot.value.ev import Selection, ValuePick


def _staked(match, sel, odds, prob, stake, fair=None) -> StakedPick:
    selection = Selection(match=match, market="1X2", selection=sel, odds=odds, bookmaker="Pinnacle")
    pick = ValuePick(selection=selection, model_prob=prob, edge=prob * odds - 1, fair_prob=fair)
    return StakedPick(pick=pick, stake=stake, full_kelly=0.1, applied_fraction=0.025)


def test_render_selection_mapea_lados():
    home = Selection("Argentina vs Mexico", "1X2", "home", 2.0)
    away = Selection("Argentina vs Mexico", "1X2", "away", 4.0)
    draw = Selection("Argentina vs Mexico", "1X2", "draw", 3.4)
    assert "Argentina" in render_selection(home) and "local" in render_selection(home)
    assert "Mexico" in render_selection(away)
    assert render_selection(draw) == "Empate"


def test_format_single_incluye_datos_clave():
    s = _staked("Argentina vs Mexico", "home", 2.0, 0.55, 2.50, fair=0.50)
    txt = format_single(s)
    assert "Argentina vs Mexico" in txt
    assert "2.00" in txt          # cuota
    assert "$2.50" in txt         # stake
    assert "+10.0%" in txt        # edge


def test_format_daily_card_con_singles_y_combinada():
    singles = [
        _staked("Argentina vs Mexico", "home", 2.0, 0.55, 2.50, fair=0.50),
        _staked("Spain vs Brazil", "home", 2.5, 0.45, 1.80, fair=0.40),
    ]
    parlay = build_parlay([singles[0].pick, singles[1].pick])
    parlays = [("Conservadora", "🔒", parlay, 1.00)]

    card = format_daily_card(singles, parlays, bankroll=100.0, date_str="14/06/2026")

    assert "CARTILLA MUNDIAL — 14/06/2026" in card
    assert "SIMPLES" in card
    assert "COMBINADAS" in card
    assert "Conservadora" in card
    assert "responsabilidad" in card  # disclaimer


def test_format_daily_card_sin_value():
    card = format_daily_card([], [], bankroll=100.0, date_str="14/06/2026")
    assert "no hay apuestas simples" in card.lower()


def test_split_message_parte_textos_largos():
    long_text = "\n".join(f"linea {i}" for i in range(2000))
    chunks = _split_message(long_text, limit=4096)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)


async def test_send_telegram_dry_run_no_necesita_token():
    ok = await send_telegram("hola", token="", chat_id="", dry_run=True)
    assert ok is True
