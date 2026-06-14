"""Formateo de la cartilla de picks para Telegram (HTML) — Agente 5.

Funciones puras (sin red): toman los picks ya stakeados y las combinadas y arman
el texto. Se testean sin tocar Telegram. Cada pick muestra el razonamiento:
probabilidad del modelo vs mercado, edge, y stake (¼ Kelly).
"""

from __future__ import annotations

from mundial_bot.staking.kelly import StakedPick
from mundial_bot.staking.parlays import Parlay
from mundial_bot.value.ev import Selection


def render_selection(sel: Selection) -> str:
    """Texto legible de la selección (ej. 'Argentina (local)', 'Empate', 'Over 2.5')."""
    home, away = (sel.match.split(" vs ", 1) + [""])[:2] if " vs " in sel.match else ("", "")
    mapping = {
        "home": f"{home} (local)",
        "away": f"{away} (visita)",
        "draw": "Empate",
        "over": f"Over {sel.market.replace('OU', '')}".strip(),
        "under": f"Under {sel.market.replace('OU', '')}".strip(),
        "yes": "Ambos marcan: SÍ",
        "no": "Ambos marcan: NO",
    }
    return mapping.get(sel.selection, f"{sel.market}:{sel.selection}")


def format_single(staked: StakedPick) -> str:
    """Formatea un pick simple."""
    pick = staked.pick
    sel = pick.selection
    lines = [
        f"⚽ <b>{sel.match}</b>",
        f"   🎯 {render_selection(sel)} @ <b>{sel.odds:.2f}</b> · {sel.bookmaker}",
    ]
    if pick.fair_prob is not None:
        lines.append(
            f"   📊 Modelo {pick.model_prob:.1%} vs mercado {pick.fair_prob:.1%} · "
            f"Edge <b>+{pick.edge:.1%}</b>"
        )
    else:
        lines.append(f"   📊 Modelo {pick.model_prob:.1%} · Edge <b>+{pick.edge:.1%}</b>")
    lines.append(f"   💵 Stake: <b>${staked.stake:.2f}</b> (¼ Kelly)")
    return "\n".join(lines)


def format_parlay(parlay: Parlay, stake: float, *, label: str, emoji: str) -> str:
    """Formatea una combinada con sus patas y métricas."""
    lines = [
        f"{emoji} <b>{label}</b> ({parlay.n_legs} patas) · "
        f"cuota <b>{parlay.combined_odds:.2f}</b> · EV <b>+{parlay.combined_ev:.1%}</b>",
    ]
    for leg in parlay.legs:
        lines.append(f"   • {render_selection(leg.selection)} @ {leg.selection.odds:.2f}")
    payout = stake * parlay.combined_odds
    lines.append(
        f"   💵 Stake: <b>${stake:.2f}</b> → paga ${payout:.2f} "
        f"(prob. {parlay.combined_prob:.1%})"
    )
    return "\n".join(lines)


def format_daily_card(
    singles: list[StakedPick],
    parlays: list[tuple[str, str, Parlay, float]],
    *,
    bankroll: float,
    date_str: str,
) -> str:
    """Arma la cartilla completa del día.

    Args:
        singles: picks simples ya stakeados.
        parlays: lista de (label, emoji, Parlay, stake).
        bankroll: banca actual.
        date_str: fecha legible (ej. "14/06/2026").
    """
    exposure = sum(s.stake for s in singles) + sum(stk for *_, stk in parlays)
    out = [
        f"🏆 <b>CARTILLA MUNDIAL — {date_str}</b>",
        f"💰 Banca: ${bankroll:.2f} · Exposición hoy: ${exposure:.2f}",
        "",
    ]

    if singles:
        out.append("<b>━━━ SIMPLES (value) ━━━</b>")
        out.extend(format_single(s) for s in singles)
    else:
        out.append("Hoy no hay apuestas simples de valor. 🤷 Mejor no forzar.")

    if parlays:
        out.append("")
        out.append("<b>━━━ COMBINADAS ━━━</b>")
        out.extend(
            format_parlay(par, stk, label=label, emoji=emoji)
            for label, emoji, par, stk in parlays
        )

    out.append("")
    out.append("⚠️ <i>El bot sugiere; vos confirmás y apostás. Apostá con responsabilidad.</i>")
    return "\n".join(out)
