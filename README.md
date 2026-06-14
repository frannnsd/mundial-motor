# ⚽ mundial-value-bot

Bot multi-agente de **value betting** para el Mundial 2026. No "adivina ganadores":
calcula la probabilidad real de cada partido con un modelo estadístico, la compara
contra las cuotas de la casa, y **solo sugiere apuestas donde la casa se equivocó en el precio** (+EV).
Entrega los picks por Telegram con su razonamiento y trackea el rendimiento de forma honesta.

> ⚠️ **El bot sugiere, vos apostás manual.** Nunca automatiza mover plata real.

---

## La realidad (leé esto antes que nada)

- La casa tiene un margen (~5-10%) metido en cada cuota. El único edge real es el **value betting**:
  apostar cuando *nuestra* probabilidad calculada supera la probabilidad implícita de la cuota.
- Las **combinadas** multiplican el margen → casi siempre son **-EV**. El bot las arma pero te muestra
  el EV real de cada una. Solo tienen valor cuando las patas están **correlacionadas** (ej. mismo partido)
  y la casa las precia como independientes.
- Con $100 y varianza alta, el crecimiento es lento. Usamos **¼ Kelly** + topes de exposición para no fundirnos.
- Métrica de verdad: **CLV (Closing Line Value)** — ¿le ganamos al precio de cierre del mercado?
  Si ganás plata pero tu CLV es negativo, es suerte y va a revertir.

---

## Arquitectura (6 agentes)

```
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
│ 1. COLECTOR  │──▶│ 2. MODELADOR │──▶│ 3. CAZADOR DE   │
│ resultados   │   │ Elo +        │   │    VALUE        │
│ Elo, fixtures│   │ Dixon-Coles  │   │ de-vig + EV     │
│ xG StatsBomb │   │ → prob. real │   │ → marca +EV     │
└──────────────┘   └──────────────┘   └────────┬────────┘
                                                │
        ┌───────────────┐   ┌──────────────┐   │
        │ 6. LEDGER     │◀──│ 5. NOTIFIC.  │◀──┤
        │ CLV/ROI/Brier │   │ Telegram     │   │
        │ (SQLite)      │   └──────────────┘   ▼
        └───────────────┘            ┌──────────────────┐
                                     │ 4. GESTOR RIESGO │
                                     │ ¼ Kelly + topes  │
                                     │ singles+combinad.│
                                     └──────────────────┘
```

| # | Agente | Módulo | Función |
|---|--------|--------|---------|
| 1 | Colector | `collectors/` | Históricos (martj42), Elo (eloratings), fixtures (API-Football), xG (StatsBomb) |
| 2 | Modelador | `models/` | Elo base + Dixon-Coles → prob. de 1X2, over/under, BTTS, marcador |
| 3 | Cazador de value | `value/` | De-vig (Shin) de las cuotas + cálculo de EV → marca los +EV |
| 4 | Gestor de riesgo | `staking/` | ¼ Kelly sobre la banca + topes; arma singles y combinadas con su EV |
| 5 | Notificador | `notify/` | Telegram (aiogram + APScheduler) con razonamiento y objetivo de CLV |
| 6 | Ledger | `ledger/` | Registra cada pick/resultado/CLV/ROI/Brier (SQLite) |

---

## Roadmap

- [ ] **Fase 1 — Datos** (`collectors/`): resultados históricos + Elo. *(en curso)*
- [ ] **Fase 2 — Modelo** (`models/`): Elo baseline → Dixon-Coles con time-decay, regularizado con priors de Elo.
- [ ] **Fase 3 — Value** (`value/`): cliente The Odds API + de-vig Shin/power + cálculo de EV.
- [ ] **Fase 4 — Staking** (`staking/`): ¼ Kelly, topes, singles + combinadas correlacionadas.
- [ ] **Fase 5 — Backtesting** : walk-forward, CLV, ROI, Brier/log-loss. Validar que el edge es real.
- [ ] **Fase 6 — Telegram** (`notify/`): envío diario programado de la cartilla de picks.
- [ ] **Fase 7 — Ledger + Deploy** : tracking en vivo + deploy cloud 24/7.

> Hallazgo clave de la investigación: en datos de selecciones (poca data), un **Elo bien afinado
> (~60% acierto, RPS 0.171) iguala o supera a Dixon-Coles y a modelos de ML**. Por eso Elo es la base
> y Dixon-Coles se usa para mercados de goles. La solución al problema de poca data: priors de Elo +
> regularización fuerte + blending de xG de clubes.

---

## Stack

| Capa | Herramienta |
|------|-------------|
| Modelo / de-vig | [`penaltyblog`](https://github.com/martineastwood/penaltyblog) (Dixon-Coles + Shin) |
| Históricos | [`martj42/international_results`](https://github.com/martj42/international_results) (CSV, ~50k partidos) |
| Elo | [`datafc`](https://pypi.org/project/datafc/) / eloratings.net |
| Fixtures vivo | [API-Football](https://www.api-football.com) (100 req/día gratis) |
| Cuotas | [The Odds API](https://the-odds-api.com) (500 créditos/mes gratis) |
| Kelly | [`keeks`](https://github.com/wdm0006/keeks) |
| xG | [StatsBomb open-data](https://github.com/statsbomb/open-data) (`statsbombpy`) |
| Telegram | [`aiogram`](https://github.com/aiogram/aiogram) + APScheduler |

---

## Setup

```powershell
# 1. Entorno (ya creado: .venv con Python 3.14)
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,model,xg]"

# 2. Claves
copy .env.example .env   # y completá ODDS_API_KEY, API_FOOTBALL_KEY, TELEGRAM_*

# 3. Tests
pytest
```

## Claves que necesitás sacar

1. **The Odds API** → https://the-odds-api.com (email, gratis, al instante) → `ODDS_API_KEY`
2. **API-Football** → https://dashboard.api-football.com (gratis) → `API_FOOTBALL_KEY`
3. **Telegram** → hablá con [@BotFather](https://t.me/BotFather), `/newbot` → `TELEGRAM_BOT_TOKEN`;
   tu chat id con [@userinfobot](https://t.me/userinfobot) → `TELEGRAM_CHAT_ID`
