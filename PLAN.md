# 🎯 Plan maestro — De "muy bueno" a "lo mejor posible"

Objetivo: dejar el bot al máximo para usarlo todo lo que resta del Mundial 2026,
aprovechando a full la API-Football **Pro** (7500 req/día) y el loop de autoaprendizaje.

## La verdad primero (techo realista)

"Inmejorable" es la meta, pero hay un techo: las casas tienen ejércitos de
cuantitativos y rara vez se les gana al cierre. Lo que SÍ podemos lograr: un bot
**muy preciso, calibrado, que aprende solo y usa más información que la mayoría**.
No infalible — sí, de primer nivel y honesto con sus probabilidades.

## Estado actual (base sólida ya construida)

- Ganador/goles: Elo internacional (validado, RPS 0.174) + Dixon-Coles, **autoalimentado** con resultados del Mundial.
- Córners/tarjetas: Poisson/Negative-Binomial con forma reciente (355 partidos) + shrinkage. **Sin validar todavía.**
- Loop de autoevaluación: loguea → califica contra resultados reales → balance (/balance). Validado: 12/20.
- Bot conversable (@Apu2000_bot) + envío diario.

---

## FASE 1 — Validar (saber qué funciona) 🔴 prioridad

> No se optimiza lo que no se mide.

1. **Backtest de córners y tarjetas** (walk-forward, como goles): RPS/Brier/acierto. Saber si tienen edge real.
2. **Backtest multi-mercado** y reporte de fiabilidad por mercado.
3. **Calibración inicial** sobre el histórico del backtest (no esperar al track record en vivo).

## FASE 2 — Exprimir la API Pro (los datos que NO usamos) 🔴 alto impacto

4. **Lesiones y suspensiones** (`/injuries`) — si falta un titular clave, ajustar la fuerza del equipo.
5. **Alineaciones** (`/fixtures/lineups`, ~30-40 min antes) — XI confirmado → re-predicción pre-partido.
6. **Recencia**: agregar fecha a `team_stats` y **pesar más los partidos recientes** (decaimiento exponencial).
7. **Tiros como feature de córners** (ya bajo `shots`, no lo exploto) + **xG** para goles.
8. **Head-to-head** (`/fixtures/headtohead`) — tendencias del cruce.
9. **Standings/forma** del grupo — momentum.

## FASE 3 — Capa de noticias / contexto (idea de Franco) 🟠

10. **Núcleo confiable = lesiones/alineaciones estructuradas** (Fase 2, sin scraping).
11. **Capa LLM de noticias** (opcional, necesita keys): NewsAPI/GNews por equipo + **Claude** para extraer señales
    (lesión de crack, ánimo, cambio táctico, presión) → ajuste cualitativo. Más ruidoso; se usa como complemento, no como verdad.

## FASE 4 — Mejor modelado 🟠

12. **Blend Elo + Dixon-Coles** para 1X2.
13. **Ensemble**: combinar señales (Elo, DC, forma, H2H, lesiones, noticias) con pesos afinados en backtest.
14. **Fix equipos chicos** (Curaçao en 1500): prior desde ranking FIFA / Elo inicial sembrado.
15. **Tuneo de parámetros** (xi del Dixon-Coles, K del shrinkage, dispersión NB) por backtest.

## FASE 5 — Calibración viva + UX del bot 🟡

16. **Auto-calibración** en vivo (Platt/isotónica por mercado) cuando el track record alcance.
17. **Bot más rico**: el "por qué" de cada pick (factores clave), resumen de noticias del partido, niveles de confianza.
18. (Opcional) **Conversación con Claude** para preguntas libres usando los datos del modelo como contexto.

## FASE 6 — Operación 24/7 🟢

19. **Deploy en la nube** (Dockerfile listo): chat bot siempre activo + cron diario.
20. **Disparadores**: update pre-partido (cuando salen las alineaciones) + grading post-partido.
21. **Monitoreo** y manejo de errores.

---

## Orden de ejecución para esta noche

Validar (F1) → datos Pro: lesiones + recencia + tiros (F2) → fix equipos chicos + blend (F4) →
calibración (F5) → noticias si hay keys (F3) → deploy (F6).

## Decisiones abiertas

- **Capa de noticias LLM**: ¿Franco pone su `ANTHROPIC_API_KEY` (+ una news API gratis)? Si no, usamos solo lesiones/alineaciones estructuradas (igual muy potente).
- Presupuesto de requests: 7500/día alcanza de sobra para todo esto con cache.
