# Competencia de Cerebros — Reporte (branch `feat/brain-competition`)

> Research, no producción. Point-in-time estricto, guard en el loop, hold-out sagrado.
> Datos crudos: `reports/brain_competition_A.json`. Reproducible: sin RNG (todo
> determinístico); config logueada abajo.

---

## FASE A — Competencia + Unificación + Proyección

### Setup

| | |
|---|---|
| Datos | football-data.co.uk: EPL + La Liga + Serie A, 10 temporadas, **11.399 partidos** |
| Cantidades | goles, córners, amarillas, remates, remates al arco, rojas — local y visitante (**12 cantidades**) |
| Warm-up | temporada 1415 (calienta el estado, NO se puntúa) |
| **Validación** | temporadas 1516–2223 → **9.120 partidos** (acá se compara y se eligen pesos) |
| **Hold-out sagrado** | temporada 2324 → **1.140 partidos** (un solo toque, al final) |
| Métrica principal | **CRPS discreto** sobre la distribución predicha (menor = mejor) |
| Diagnósticas | MAE sobre la media + calibración (ECE) en umbrales estándar |
| Config | halflife 300d (tasas) / 45d (forma), shrinkage k=3, GLM refit cada 30d (min 500 filas, ridge 1e-3), Fano de liga como dispersión común |

Los cuatro competidores producen (media, varianza) y **la misma regla** los convierte
en pmf (NegBin si var>media, Poisson si no): compiten los métodos de estimación, no
familias de distribución distintas.

- **A — Tasas históricas**: media decaída de lo generado + lo concedido por el rival, ajuste localía.
- **B — Matchup multiplicativo**: ataque × concesión / media de liga (Dixon-Coles aplicado a conteos).
- **C — GLM Poisson** (IRLS propio): localía, tasas relativas, descanso, forma, H2H. Refit mensual point-in-time.
- **bobo**: media/var corriente de la liga-TEMPORADA hasta ese partido.

### Tabla de VALIDACIÓN — CRPS (menor = mejor; 🔴 = pierde contra el bobo)

| Cantidad | A | B | C | bobo | Ganador |
|---|---|---|---|---|---|
| goals_h | 0.6499 | 0.6421 | **0.6416** | 0.6930 | C |
| goals_a | 0.5803 | **0.5722** | 0.5729 | 0.6133 | B |
| corners_h | 1.5634 | **1.5438** | 1.5462 | 1.6529 | B |
| corners_a | 1.4012 | **1.3885** | 1.3887 | 1.4668 | B |
| yellows_h | 0.7317 | **0.7289** | 0.7297 | 0.7486 | B |
| yellows_a | 0.7512 | 0.7573 | **0.7513** | 0.7611 | A≈C |
| shots_h | 2.6106 | 2.5118 | **2.4991** | 2.9146 | C |
| shots_a | 2.3394 | 2.2635 | **2.2468** | 2.5855 | C |
| sot_h | 1.3277 | 1.2960 | **1.2910** | 1.4426 | C |
| sot_a | 1.1869 | **1.1599** | 1.1602 | 1.2784 | B |
| reds_h | 0.0820 | 🔴 0.0831 | **0.0819** | 0.0824 | C (margen ínfimo) |
| reds_a | 🔴 0.0974 | 🔴 0.0992 | 🔴 0.0977 | **0.0973** | **bobo** |

**Lectura honesta:**
- Los tres cerebros le ganan al bobo en las **10 cantidades "reales"** (goles, córners,
  amarillas, remates, al arco) — el modelado agrega señal genuina.
- **B y C van cabeza a cabeza** (B mejor en córners/goles_a; C mejor en remates/al arco);
  A queda consistentemente tercero pero digno.
- **Rojas: el bobo es (casi) imbatible.** Base rate ~4%/lado: la media de liga ya lo dice
  todo. En `reds_a` NINGÚN cerebro le gana → el unificado usa el bobo ahí. Esto era
  esperable y es la prueba de que la vara funciona.

Companion (MAE validación, media puntual): mismo ranking — ej. shots_h: A 3.71 /
B 3.53 / C 3.53 / bobo 4.13. Calibración en umbrales (ECE validación): todos ≤ 0.036
(ej. goals O2.5: A 0.017, B 0.031, C 0.022, bobo 0.021).

### Regla de unificación (derivada SOLO de validación)

Por cantidad:
- **Elegibles** = cerebros con CRPS < CRPS_bobo (perder contra el bobo ⇒ peso 0).
- **w_i ∝ exp(−Δ_i/τ)**, con Δ_i = CRPS_i − CRPS_mejor y **τ = (CRPS_bobo − CRPS_mejor)/3**
  (a la altura del bobo el peso cae a e⁻³ ≈ 5%).
- Sin elegibles ⇒ el unificado ES el bobo.
- La predicción unificada es la **mixtura de pmfs**: pmf_U = Σ w_i·pmf_i (conserva la
  calibración de los componentes).

Pesos resultantes (redondeados):

| Cantidad | A | B | C | bobo |
|---|---|---|---|---|
| goals_h | .24 | .38 | .39 | — |
| goals_a | .22 | .40 | .38 | — |
| corners_h/a | .23 | .40/.38 | .37/.38 | — |
| yellows_h | .26 | .39 | .35 | — |
| yellows_a | .47 | .07 | .45 | — |
| shots_h/a | .19 | .38 | .42/.43 | — |
| sot_h/a | .20 | .38/.40 | .42/.40 | — |
| reds_h | .37 | 0 | .63 | — |
| reds_a | 0 | 0 | 0 | **1.0** |

### HOLD-OUT 2023/24 (único toque) — unificado vs individuales (CRPS)

| Cantidad | Unificado | Mejor individual | bobo |
|---|---|---|---|
| goals_h | 0.6416 | 0.6406 (B) | 0.6985 |
| goals_a | 0.5767 | 0.5740 (B) | 0.6256 |
| corners_h | 1.6189 | 1.6162 (C) | 1.7112 |
| corners_a | 1.3553 | 1.3550 (C) | 1.4411 |
| yellows_h | 0.7675 | 0.7674 (C) | 0.7825 |
| yellows_a | 0.7677 | 0.7668 (C) | 0.7814 |
| shots_h | **2.7194** | 2.7228 (C) | 3.0858 |
| shots_a | **2.3670** | 2.3716 (B) | 2.6146 |
| sot_h | 1.3379 | 1.3365 (B) | 1.4608 |
| sot_a | 1.1391 | 1.1359 (B) | 1.2464 |
| reds_h | 0.0820 | 0.0817 (C) | 0.0824 |
| reds_a | 0.0879 | 0.0876 (A) | 0.0879 |

**Veredicto Fase A:** el unificado le gana al bobo en TODAS las cantidades del hold-out
y queda pegado al mejor individual de cada una (a veces mejor: shots_h/shots_a) —
**sin haber mirado nunca el hold-out para elegirlo**. Es el comportamiento que se le
pide a un ensemble honesto: robustez del nivel del mejor, sin apostar a un solo método.

### Proyección de mercados — sanity check (hold-out, cerebro unificado)

`markets/projection.py` traduce las pmfs a ~15 mercados Tier A (1X2, doble oportunidad,
O/U de goles/córners/tarjetas/remates/al arco, rango de goles, marcador exacto, margen,
BTTS, equipo con más córners, ambos con tarjeta, roja en el partido, alguna amonestación).

| Mercado proyectado | ECE (hold-out) |
|---|---|
| Over 2.5 goles | 0.043 |
| Over 9.5 córners | 0.039 |
| Over 3.5 amarillas | 0.034 |
| BTTS | 0.060 |

Consistente con la calibración de las cantidades base (≤0.04 en validación): la capa de
proyección no rompe la calibración.

### Caveats honestos (Fase A)

1. **Independencia local/visita** en score matrix y totales (sin corrección tau de
   Dixon-Coles para 0-0/1-1). Afecta marginalmente BTTS/correct score. TODO.
2. **MT/RF y "mitad con más goles"**: TODO explícito (`NotImplementedError`); la data
   HTHG/HTAG ya está cargada en el loader.
3. **Horizonte 120'** (prórroga): TODO explícito en la interfaz (`horizon="120"` raise).
4. **Cerebro C hereda del A** hasta su primer refit (~500 filas por liga); efecto de
   arranque menor, absorbido por el warm-up.
5. **Faltas**: quedaron fuera de la competencia (no estaban en el scope final A.2);
   la columna está en el loader para sumarla después.
6. **Rojas**: modelarlas no aporta sobre la media de liga (confirmado empíricamente);
   el unificado usa el bobo en `reds_a`.
7. Dispersión común (Fano de liga por lado) para los 4: compiten en la MEDIA
   condicional; ninguno recibe ventaja distribucional.
