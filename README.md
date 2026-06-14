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

## Roadmap — estado

- [x] **Fase 1 — Datos** (`collectors/`): resultados históricos (descarga real de martj42).
- [x] **Fase 2 — Modelo** (`models/`): Elo internacional + Dixon-Coles con time-decay y neutral_venue.
- [x] **Fase 3 — Value** (`value/`): cliente The Odds API + de-vig Shin + cálculo de EV.
- [x] **Fase 4 — Staking** (`staking/`): ¼ Kelly con topes + combinadas (conservadora y alto riesgo).
- [x] **Fase 5 — Backtesting** (`backtest/`): walk-forward out-of-sample (RPS/Brier/log-loss).
- [x] **Fase 6 — Telegram** (`notify/`): formato de cartilla + envío + scheduler diario.
- [x] **Fase 7 — Ledger + Deploy** (`ledger/`, `pipeline.py`, `Dockerfile`): tracking + deploy cloud.
- [ ] **Pendiente (necesita keys):** correr en vivo con cuotas reales + medir CLV; afinar `xi`/draw-model.

### Validación del modelo (backtest real, out-of-sample)

Walk-forward del Elo sobre **10.999 partidos (2015→2026)**:

| Métrica | Valor | Referencia |
|---------|-------|------------|
| RPS | **0.174** | bueno ≈ 0.17-0.19; azar ≈ 0.22 |
| Accuracy (1X2) | **59.8%** | azar ≈ 33% |

El modelo predice con poder real (a la par de los papers publicados). ⚠️ Buen RPS
≠ ganancia garantizada: el test definitivo es el **CLV** con cuotas en vivo.

> Corré la cartilla de ejemplo sin ninguna API key:
> `python scripts/run_daily.py --sample`

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

## Cómo correr

```powershell
python scripts/fetch_statsbomb.py        # baja datos de córners/tarjetas (1 vez, ~min)
python scripts/test_telegram.py          # 🔌 prueba la conexión con Telegram (token + chat_id)
python scripts/predict_matches.py        # ⭐ reporte multi-mercado por partido (Telegram)
python scripts/predict_matches.py --schedule  # envía las predicciones a diario
python scripts/run_daily.py --sample     # cartilla de value betting (singles + combinadas)
python scripts/backtest_elo.py           # backtest del modelo de goles/ganador
pytest -m "not network"                  # 79 tests
```

### Conectar Telegram (3 pasos)

1. [@BotFather](https://t.me/BotFather) → `/newbot` → copiá el **token** → `TELEGRAM_BOT_TOKEN` en `.env`.
2. [@userinfobot](https://t.me/userinfobot) → copiá tu **id** numérico → `TELEGRAM_CHAT_ID` en `.env`.
   Abrí un chat con tu bot nuevo y mandale `/start` (si no, no te puede escribir).
3. Corré `python scripts/test_telegram.py` → si todo está bien, te llega un mensaje de prueba. ✅

### El predictor multi-mercado (lo principal)

Por cada partido te dice lo más probable en **ganador, goles, córners, tarjetas y
ambos marcan**, con la **cuota justa** de cada uno:

```
⚽ Argentina vs Mexico
   🏆 Gana: Argentina (64%) · justo @ 1.55
   ⚽ Goles: ~1.8 → Over 1.5 goles (55%) · justo @ 1.80
   🚩 Córners: ~8.7 → Under 8.5 córners (50%) · justo @ 1.99
   🟨 Tarjetas: ~4.0 → Under 4.5 tarjetas (62%) · justo @ 1.60
   🤝 Ambos marcan: No (69%) · justo @ 1.46
```

Modelos: Elo (ganador) · Dixon-Coles (goles/BTTS) · córners y tarjetas (Poisson sobre
datos de StatsBomb: WC 2018/2022, Euro, Copa América, AFCON — 314 partidos, 100 árbitros).

## Casas de apuestas (Argentina · Prov. de Buenos Aires)

Las cuotas de córners/tarjetas de las casas `.bet.ar` **no están en ninguna API barata**
(los feeds que las tienen son enterprise, ~US$5.000/mes). Por eso el flujo es:

> **El bot te da la cuota justa → vos la comparás en tu casa → si paga más, apostás.**

Casas recomendadas (todas con goles + córners + tarjetas, toman pesos):
- **Bet365** y **bplay** — abrí las dos para comparar líneas (en córners/tarjetas varían mucho).
- **Stake** (`pba.stake.bet.ar`) — si querés cripto; ahora con licencia en Prov. de Buenos Aires.

## Deploy cloud 24/7 (Railway)

1. Subí el repo a GitHub y conectalo en [Railway](https://railway.app) (detecta el `Dockerfile`).
2. Cargá las variables de entorno (las mismas del `.env`) en el panel de Railway.
3. Montá un **volumen** en `/app/data` para que la banca y el ledger persistan entre deploys.
4. El contenedor corre `run_daily.py --schedule` y manda la cartilla todos los días.

> También hay `Procfile` (Render/Heroku) y `railway.json` con la config de arranque.
