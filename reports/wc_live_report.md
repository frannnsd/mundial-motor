# Mundial en marcha — Reporte (branch `feat/wc-live`)

> Reentreno sobre SELECCIONES + validación contra el torneo jugado + horizonte 120'
> + pipeline diario. Research → operación. Point-in-time estricto, guard en el loop.
> Datos crudos: `reports/wc_validation.json`. NO mergeado a main.

---

## PARTE 1 — Data de selecciones (API-Football)

| | |
|---|---|
| Partidos con stats | **1.610** (2022-01-09 → 2026-07-01), 177 selecciones |
| Por tipo | eliminatoria 554 · amistoso 416 · continental 262 · nations_league 234 · **mundial 144** (Qatar 2022 completo + WC 2026 jugado, 100% con stats) |
| Llamadas API | **161 de 1.200 presupuestadas** (batches de 20 ids, cache-primero: re-correr = 0 llamadas, ~60MB en `data/nt_cache/`) |
| STOP-check | ✅ aprobado — ninguna selección viva con <10 partidos (mínimos: Cape Verde 18, Ghana 27, Algeria 30) |
| Sin stats (excluidos, NO rellenados) | Friendlies 244/660, AFCON-Qual 112, WC-Qual África 57, WC-Qual Asia 18, torneos menores |

Aproximaciones documentadas: "terminado" = {FT, AET, PEN} (en AET los goles son de 90'
pero las stats cubren 120' — inconsistencia menor conocida); `neutral` = mundial salvo
anfitrión (Qatar'22; USA/Mex/Can'26), continental → True, resto → False.

## PARTE 2 — Adaptaciones selecciones ≠ clubes (mismos cerebros)

1. **Shrinkage más fuerte:** k=8 (vs 3 en clubes) — `(n·tasa + 8·media)/(n+8)`.
2. **Decay:** tuneado en validación → **halflife 365 días** ganó (grilla abajo).
3. **Peso por match_type:** amistoso **0.75** (tuneado), oficial 1.0, mundial/continental 1.2, otro 0.8.
4. Bobo = promedio de internacionales **competitivos** point-in-time (sin amistosos).
5. Cancha neutral: sin ajuste de localía; bobo agrupa ambos lados; GLM con flag 0.5.

Grilla de tuning (score = CRPS/bobo promedio de los 3 cerebros × 10 cantidades; SOLO validación):

| halflife | amistoso | score |
|---|---|---|
| 365 | 0.5 | 0.91785 |
| **365** | **0.75** | **0.91218** ← elegido |
| 730 | 0.5 | 0.92189 |
| 730 | 0.75 | 0.91701 |

## PARTE 3 — Validación contra el torneo jugado (81 partidos)

Setup legítimo: estado entrenado con TODO el histórico ANTERIOR a cada kickoff
(as_of=kickoff, guard en el loop); los partidos previos del propio Mundial alimentan
los siguientes (realista). **81 = 72 de grupos + 9 de eliminatoria ya jugados.**

CRPS (menor = mejor; 🔴 pierde vs bobo):

| Cantidad | A | B | C | bobo |
|---|---|---|---|---|
| goals_h | 0.8315 | **0.8177** | 0.8314 | 0.8893 |
| goals_a | 0.5457 | 0.5090 | **0.5056** | 0.5872 |
| corners_h | 1.7509 | 1.7177 | **1.6876** | 1.8248 |
| corners_a | 1.3423 | 1.3059 | **1.3022** | 1.4643 |
| yellows_h | 0.5996 | 0.5840 | **0.5837** | 0.6091 |
| yellows_a | 0.6295 | 0.6097 | **0.6084** | 0.6833 |
| shots_h | 3.5781 | 3.4698 | **3.4439** | 3.7673 |
| shots_a | 2.9004 | 2.7740 | **2.7609** | 3.2179 |
| sot_h | 1.5820 | 1.5291 | **1.5273** | 1.7151 |
| sot_a | 1.2292 | 1.1732 | **1.1659** | 1.3322 |
| reds_h | 🔴 0.0378 | 🔴 0.0402 | 🔴 0.0370 | **0.0367** |
| reds_a | 0.0915 | 0.0910 | 0.0919 | 0.0940 |

**Los 3 cerebros le ganan al bobo en las 10 cantidades reales, también en selecciones.**
C levemente puntero (córners/remates/al arco), B en goles del local.

### Unificación NT (pesos SOLO de esta validación — no se reusan los de clubes)

Regla: perdedor vs bobo ⇒ 0; si el gap relativo entre elegibles < 1.5% ⇒ **uniformes**
(no se sobreinterpretan 81 partidos); si no, softmax inverso-CRPS.

| Cantidad | A | B | C | bobo |
|---|---|---|---|---|
| goals_h | .26 | .47 | .27 | — |
| goals_a | .11 | .42 | .47 | — |
| corners_h/a | .14/.20 | .29/.39 | .57/.42 | — |
| yellows_h/a | .07/.18 | .45/.40 | .47/.42 | — |
| shots_h/a | .14/.17 | .38/.40 | .48/.43 | — |
| sot_h/a | .18/.15 | .41/.40 | .42/.46 | — |
| reds_h | — | — | — | **1.0** |
| reds_a | **.333** | **.333** | **.333** | — (regla uniforme activada) |

**CAVEAT OBLIGATORIO:** 81 partidos es una muestra CHICA — los pesos son ruidosos por
naturaleza. La regla uniforme mitiga; los 81 son VALIDACIÓN (eligen pesos), no hold-out.
**El hold-out real del sistema de selecciones es el forward-test de la fase
eliminatoria**, que se acumula solo con el pipeline diario.

## PARTE 4 — Horizonte 120' + mercados de eliminatoria

- **Prórroga condicional:** p_et = P(empate a 90'); pmf_TE = (1−p_et)·pmf₉₀ + p_et·(pmf₉₀ ⊛ Poisson(μ₉₀·⅓·0.9)). FATIGUE=0.9 **fijo y documentado** (no se tunea con 81 partidos).
- **"Se clasificará"**: P(gana 90') + P(empate)·[P(gana ET) + P(empate ET)·P(penales)]. **Penales 50/50** default honesto.
- **"Método de victoria"**: reglamentario / prórroga / penales por equipo (suma 1, testeado).
- Totales TE (goles/córners/tarjetas/remates/al arco) + ambos-con-tarjeta TE.
- **Coherencia de props en TE testeada**: `team_total_at_horizon(μ₉₀, p_et)` alimenta el
  reparto por jugador y Σ jugadores == total del equipo en el MISMO horizonte.
- Los mercados 1X2/marcador/margen/BTTS siguen liquidando a 90' (convención bet365);
  pedirles 120' sigue levantando error explícito.

## PARTE 5 — Pipeline diario de operación (`python -m mundial_bot.wc.daily`)

| Comando | Qué hace | Llamadas API |
|---|---|---|
| `pre-day [--date]` | Reporte del día (`reports/daily/FECHA.md`): cantidades del unificado NT + mercados 90' + eliminatoria/TE si aplica + props con XI probable. **Registra todo lo emitido** en el forward-test (as_of en notes) | 1-2 (lista del día; motor y props usan cache) |
| `pre-kickoff --fixture N` | Baja lineups SOLO en la ventana −60min→kickoff, recalcula props con XI confirmado, imprime el **delta** y loguea con nota `xi_confirmado` | 1-2 |
| `post-day [--date]` | Liquida contra los resultados reales: props por jugador (`settle_fixture`) + mercados de equipo (`settle_team_fixture` con totales del fixture cacheado). Los fixtures nuevos quedan cacheados → el próximo pre-day ya los usa | batcheado + cache |
| `weekly` | Reporte acumulado del forward-test (Brier por mercado, conteos) → `reports/weekly_forward_test.md` | 0 |
| `add-odds --fixture --player --market --line --odds` | Carga manual de la línea/cuota de bet365 en una predicción logueada | 0 |

Registro: SQLite `data/forward_test.sqlite` (idempotente: la PRIMERA predicción por
(fixture, jugador, mercado) es la que vale — no se reescribe la historia). Mercados de
equipo con prefijo `team_` y player_id=0.

### Ejemplo REAL (pre-day corrido hoy, Round of 32)

**Spain vs Austria** — estado point-in-time con todo lo jugado hasta anoche:

- Cantidades (90'): goles 1.72–1.30 · córners 4.7–3.4 · amarillas 1.6–1.9 · remates 14.0–10.4 · al arco 4.7–3.8
- 1X2: **46% / 23% / 32%** · BTTS 48% · O2.5 53% · O9.5 córners 33%
- **Eliminatoria** (P prórroga 23%): se clasifica **Spain 58% / Austria 42%** · método: 90' 46/32, ET 7/5, penales 6/6
- Totales TE: O2.5 goles 57% · O9.5 córners 38% · O3.5 amarillas 51%
- Props (XI probable, horizonte 120'): Oyarzabal μ3.07 remates · P(anota) 48% · Yamal μ1.75 / 28% · Pedri P(tarjeta) 35% · Sabitzer μ1.62 / P(tarjeta) 32% · Arnautović P(anota) 30%

Reporte completo del día: `reports/daily/2026-07-02.md`. Todas las predicciones del día
quedaron logueadas en el forward-test (equipo + top-5 props por equipo por partido).

## QUOTA API — consumida y restante

| Concepto | Llamadas |
|---|---|
| P1 histórico selecciones (48 equipos, 2.238 fixtures batcheados) | 161 |
| Sondas (batch/coverage) | ~4 |
| Pipeline pre-day real de hoy | ~2 |
| **Total del programa wc-live** | **~167 de 3.000 presupuestadas (40% de 7.500/día)** |
| Costo diario del pipeline en operación | ~4-6 llamadas/día (pre-day + pre-kickoff + post-day) |

## Cómo operar lo que queda del Mundial

```
# a la mañana (predicciones + registro):
python -m mundial_bot.wc.daily pre-day
# ~30-60 min antes de cada partido (XI confirmado):
python -m mundial_bot.wc.daily pre-kickoff --fixture <id>
# a la noche (liquidación + actualización):
python -m mundial_bot.wc.daily post-day
# los domingos:
python -m mundial_bot.wc.daily weekly
# cuando cargues cuotas de bet365 a mano:
python -m mundial_bot.wc.daily add-odds --fixture <id> --player <id|0> --market shots --line 1.5 --odds 1.85
```

## Caveats honestos del programa

1. **81 partidos de validación** — pesos ruidosos por naturaleza (regla uniforme mitiga).
   El hold-out real es el forward-test de la eliminatoria, que se acumula solo.
2. **FATIGUE 0.9 y penales 50/50 son constantes documentadas**, no estimadas (no hay data).
3. En partidos AET del histórico, los goles son de 90' pero las stats cubren 120'
   (inconsistencia menor de la fuente, documentada en nt_data).
4. Independencia local/visita en el score matrix (tau DC pendiente desde la Fase A).
5. Amistosos sin stats (416 con stats de 660): los sin-stats se EXCLUYEN, no se imputan.
6. El agente P5 original murió por límite de sesión sin escribir código; el pipeline lo
   escribió el orquestador y está cubierto por 7 tests propios + auditoría guardian.
