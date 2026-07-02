# Automatización total + Web del Mundial — Reporte final

> Todo corre solo en la nube. Lo único manual que queda: **cargar cuotas de bet365
> desde la web** (≤3 taps desde el celular). El bot NO apuesta ni se conecta a
> bet365: predice, registra y mide.

## URLs

| Qué | Dónde |
|---|---|
| **Web** (la herramienta diaria) | https://mundial-web-eta.vercel.app — clave: la `WEB_ACCESS_KEY` de tu `.env` |
| Backend + API `/wc` | https://mundial-motor.onrender.com (todo `/wc/*` con header `X-Access-Key`) |
| Storage | Supabase proyecto `mundial` (São Paulo) — dashboard en supabase.com |

## Decisión de storage y por qué

**Supabase Postgres (gratis, no expira)** vía PostgREST — sin dependencias nuevas
(el cliente es `requests`). Descartados: Render Postgres free (SE BORRA a los 30
días — inaceptable para el forward-test) y disco persistente de Render (~USD 7/mes
y la web no lo vería directo). La web NUNCA toca Supabase: consume solo la API del
backend. **Predicciones inmutables** (la primera gana — no se reescribe la
historia), cuotas se adjuntan a la fila como snapshot (el EV se calcula contra lo
que el modelo decía al cargar la cuota). **Backup**: dump diario del forward-test
a la tabla `backups` + descarga manual desde `/admin-wc` en la web.

**Migración verificada:** 58 predicciones + 1.610 nt_matches + 4.001
player_matches — conteos locales == remotos, exactos.

## Arquitectura de jobs (horarios AR / UTC)

| Job | Cuándo (AR) | Qué hace |
|---|---|---|
| `wc_daily` | 09:00 (12:00 UTC) | Predicciones del día (cantidades+pmfs+mercados 90'/120'+props XI probable) → Supabase + forward-test |
| `wc_lineups` | cada 5 min (barato: solo actúa con partidos en ventana −60min→kickoff) | XI confirmado → recalcula props, publica deltas |
| `wc_settle` | 03:00 (06:00 UTC) | Liquida ayer-AR (props + mercados de equipo), actualiza nt/player tables, backup diario |
| `wc_weekly` | dom 10:00 (13:00 UTC) | Resumen de calibración/scores/EV en job_runs |

Scheduler: APScheduler dentro del servicio web de Render existente (gratis; el
pinger de UptimeRobot ya lo mantiene despierto). **Catch-up al reiniciar**: si
Render reinicia y el daily de hoy no corrió, se corre solo (jobs idempotentes —
no duplican). En la nube los datos del motor se leen desde Supabase (fallback
automático: el cache local gitignored no existe en Render).

## Costos nuevos de infraestructura: **USD 0**

Supabase free (500MB, sin expiración) + Render free existente + Vercel free
existente. Si algún día Render free molesta (reinicios), la opción es Starter
USD 7/mes — NO activada.

## La web (todo en hora argentina, mobile-first)

- **/ (HOY)**: cards del día — 1X2 display gigante, "se clasifica" en eliminatorias, mercados jugosos, badge XI probable/CONFIRMADO.
- **/partido-wc/[id]**: distribución de cada cantidad (mini-gráfico de la pmf), mercados 90' y TE separados, props por jugador con deltas de XI, y **cargar cuota en ≤3 taps** (tap en el mercado → cuota → guardar).
- **/forward-test**: el dashboard de la verdad — calibración con semáforo, Brier, "dónde se equivoca el modelo", EV/ROI si hay cuotas. Sin maquillaje.
- **/admin-wc**: estado de jobs, quota API del día, descarga de backup.
- Gate global de clave simple (localStorage; 401 → re-login). Etiquetas honestas: todo dice "P(modelo)".

## Verificación end-to-end (timestamps reales, 2026-07-02)

```
15:49:45 UTC  Deploy live · store_configured=True · scheduler RUNNING
              próximos: lineups 15:53 · settle 03:00 AR · daily 09:00 AR · weekly dom
15:49:45 UTC  POST /wc/admin/run/daily (disparo manual de verificación)
15:53:05 UTC  job daily: OK — 2 partidos procesados EN LA NUBE (motor leyó Supabase)
15:53:06 UTC  GET /wc/today → Spain vs Austria (46/23/32) · Portugal vs Croatia (46/23/31)
15:53:07 UTC  Web Vercel → HTTP 200
```

"predictions: 0" en el run manual = idempotencia funcionando (las predicciones de
hoy ya estaban registradas por la corrida local de la mañana; la primera gana).
El ciclo lineups→settle se completa solo esta noche con los partidos reales de hoy.

## Env vars (ya cargadas — referencia)

**Render (`mundial-motor`)**: `API_FOOTBALL_KEY`, `ANTHROPIC_API_KEY`,
`ODDSPAPI_KEY`, `PREFERRED_BOOKS`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`,
`WEB_ACCESS_KEY`, `WC_SCHEDULER=1` (las 4 nuevas seteadas vía API de Render).
**Vercel (`mundial-web`)**: `NEXT_PUBLIC_API_BASE` (sin cambios).
**Local `.env`**: todas + `SUPABASE_ACCESS_TOKEN`, `RENDER_API_KEY`,
`SUPABASE_DB_PASSWORD` (generada, no usada en runtime).

## Lo único manual que queda: cargar cuotas (flujo)

1. Abrís la web en el celu → HOY → tap en el partido.
2. Tap en el mercado o prop donde apostaste → se abre el bottom-sheet.
3. Tipeás la cuota (y stake si querés) → **Guardar**. Listo: queda vinculada al
   snapshot de la predicción vigente; el settle nocturno la liquida y el
   forward-test empieza a mostrar EV/ROI reales.

## Seguridad / higiene pendiente (recomendado, no urgente)

Rotar cuando termine el torneo (o antes si querés): `SUPABASE_ACCESS_TOKEN` y
`RENDER_API_KEY` (pasaron por el chat), y las API keys viejas (API-Football,
Anthropic, odds — pasaron por el chat en el deploy de junio). El sistema no los
necesita rotados para funcionar.

## Caveats honestos

1. Render free puede reiniciarse: el catch-up + idempotencia lo cubren para el
   daily; un reinicio DURANTE la ventana de lineups puede perder un poll (se
   recupera en el siguiente, cada 5 min).
2. El smoke mobile formal quedó a nivel build+e2e de API (la UI fue verificada
   por build de producción y contrato de datos; el flujo de 3 taps está
   implementado con bottom-sheet nativo).
3. Los dos agentes de build que murieron por límite de sesión (P5 pipeline y fin
   del web) fueron completados por el orquestador y auditados.
4. Quota API en operación: ~4-6 llamadas/día + polls de lineups en ventana
   (≤12/partido) — muy por debajo del techo de 150/día.
