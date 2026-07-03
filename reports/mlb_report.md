# Cerebro MLB — Reporte (M0→M4)

> Mismo rigor que el Mundial: point-in-time con guard EN el loop, hold-out sagrado,
> misma vara (CRPS/`count_pmf`), sin RNG. Datos crudos: `reports/mlb_competition.json`.

## M0 — Data (verificada)

| | |
|---|---|
| Fuente | **MLB Stats API oficial (statsapi.mlb.com) — GRATIS, sin key** |
| Partidos | **26.489** (2015→2026 parcial), 12 llamadas (1 por temporada, cache-primero) |
| Por partido | carreras, hits, **carreras F5 exactas** (entradas 1-5 del linescore), venue, **pitcher abridor** (99.95% de cobertura histórica) |
| Cuotas | `usa-mlb` disponible en odds-api.io con la key actual |
| Nota | 2020 = temporada COVID de 60 juegos (incluida; el decay le baja peso) |

## M1 — Competencia de cerebros

Familias: carreras, hits, carreras-F5 (×local/visita = 6 cantidades).
- **A** — tasas de equipo (decay) + parque (√ por lado) + localía
- **B** — matchup con ABRIDOR: la defensa rival se mezcla 60/40 con lo permitido en
  los starts previos del pitcher (proxy point-in-time)
- **C** — GLM Poisson (IRLS): localía, tasas relativas, señal del abridor, parque, descanso
- **bobo** — media/var de la liga-temporada hasta el día

Splits: warm-up 2015 · validación 2016-2024 (**21.141 partidos**) · hold-out **2025
completa (2.430)** · 2026 = forward-test vivo (no se puntúa acá).
Grilla (solo validación): halflife ∈ {180, 270, 365} → ganó **180 días** (el béisbol
cambia rápido: rosters, brazos). starter_weight fijo 0.6 (no tuneado).

### Validación — CRPS (menor = mejor; ninguno pierde vs bobo)

| Cantidad | A | B | C | bobo | Ganador |
|---|---|---|---|---|---|
| runs_h | 1.6985 | **1.6961** | 1.6969 | 1.7189 | B |
| runs_a | 1.7300 | **1.7296** | 1.7301 | 1.7534 | B |
| hits_h | 1.8181 | 1.8151 | **1.8147** | 1.8378 | C |
| hits_a | 1.9383 | 1.9355 | **1.9329** | 1.9618 | C |
| runs_f5_h | 1.3137 | **1.3110** | 1.3115 | 1.3264 | B |
| runs_f5_a | 1.2245 | **1.2225** | 1.2226 | 1.2364 | B |

**El abridor manda:** B gana carreras y F5 (donde el pitcher pesa más), C gana hits.
**Lectura honesta:** los márgenes sobre el bobo son FINOS (~1-2% de CRPS, vs 5-12% en
fútbol) — el béisbol tiene más azar por partido y MLB es un mercado eficientísimo. La
fortaleza del modelo es la CALIBRACIÓN, no una bola de cristal.

### Pesos del unificado (softmax inverso-CRPS, SOLO validación)

| Cantidad | A | B | C |
|---|---|---|---|
| runs_h/a | .28/.33 | .38/.35 | .34/.33 |
| hits_h/a | .25/.25 | .37/.33 | .38/.43 |
| runs_f5_h/a | .24/.24 | .40/.38 | .36/.37 |

### HOLD-OUT 2025 (único toque): el unificado le gana al bobo en las 6 cantidades

| | unificado | bobo |
|---|---|---|
| runs_h / runs_a | 1.6785 / 1.7904 | 1.6956 / 1.8220 |
| hits_h / hits_a | 1.8094 / 1.9466 | 1.8260 / 1.9831 |
| runs_f5_h / f5_a | 1.2590 / 1.2167 | 1.2701 / 1.2360 |

### Calibración de mercados (hold-out 2025)

| Mercado | ECE |
|---|---|
| Total carreras O8.5 | **0.0232** |
| F5 O4.5 | **0.0217** |
| Moneyline (P gana local) | **0.0360** |

## M2 — Proyección (`markets/mlb_projection.py`)

Moneyline (empate a 9 → extras 50/50 documentado) · Run line ±1.5 · Totales
7.5-10.5 · Team totals · Hits totales · **F5 con familia propia** (ML 3-way + totales
— exacto, no un escalado). Caveat: independencia local/visita (menor que en fútbol:
las ofensivas no comparten pelota).

## M3 — Props (`players/mlb_props.py`)

- **Ks del abridor**: K/out decaído (200d) × outs esperados (shrunk), NB. Demo real
  (Reds@Brewers 2-jul): Burns μ6.99 P(o5.5) 68% · Misiorowski μ8.18 P(o5.5) 81%.
- **Hits de bateadores**: reparto COHERENTE del total del equipo (Σ 9 bateadores ==
  μ equipo, exacto) con pesos por orden de bateo y tasas point-in-time.
- **P(HR)**: el prop más ruidoso (base rate ~3%/PA); shrinkage fuerte documentado.
- 10 tests sin red; 41 llamadas (cacheadas: re-correr = 0).

## Config final (reproducible)

halflife 180d · form 40d · shrink_k 10 · starter_shrink_k 4 · starter_weight 0.6 ·
park_clip [0.80, 1.25] · GLM refit 30d ridge 1e-3 · Fano de liga como dispersión.
Pesos del unificado committeados en `data/mlb_weights.json`.
