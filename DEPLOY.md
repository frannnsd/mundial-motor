# 🚀 Deploy 24/7 (Railway)

Dejá el bot corriendo solo todo el Mundial: responde por Telegram, manda las
predicciones del día, alerta antes de cada partido y se autoalimenta — sin tu PC.

El servicio es **un solo proceso** (`scripts/run_bot.py`): bot conversable +
scheduler (ciclo diario + alertas pre-partido), todo sobre la misma base SQLite.

## Pasos (≈10 min)

### 1. Subir el repo a GitHub
```bash
cd Projects/mundial-value-bot
git remote add origin https://github.com/<tu-usuario>/mundial-value-bot.git
git push -u origin main
```
> El `.env` con tus keys **NO se sube** (está en `.gitignore`). Las cargás en Railway.

### 2. Crear el proyecto en Railway
1. Entrá a [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Elegí el repo. Railway detecta el `Dockerfile` y buildea solo.

### 3. Volumen para que persistan datos y apuestas
1. En el servicio → **Settings → Volumes** → **New Volume**.
2. Mount path: `/app/data`
   *(ahí viven la base de predicciones, tus apuestas y los caches — así no se pierden en cada deploy).*

### 4. Variables de entorno
En **Variables**, pegá las mismas del `.env`:
```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
API_FOOTBALL_KEY=...
BANKROLL_USD=100
KELLY_FRACTION=0.25
TIMEZONE=America/Argentina/Buenos_Aires
DAILY_PICKS_HOUR=10
```

### 5. Primer arranque (poblar datos)
La primera vez, en Railway → **Settings → Deploy → Custom Start Command** (temporal) o
desde la consola del servicio, corré una vez:
```bash
python scripts/fetch_team_stats.py && python scripts/run_daily_update.py
```
Esto baja la forma reciente + resultados del Mundial. Después volvé el start command al
default (`python scripts/run_bot.py`). El cron diario lo mantiene fresco solo.

### 6. Listo ✅
El bot queda escuchando 24/7. Todos los días, a `DAILY_PICKS_HOUR`, manda el balance +
las predicciones; y ~1-2h antes de cada partido, la alerta pre-partido con las bajas.

## Alternativas
- **Render / Fly.io**: mismo Dockerfile, montá un disco en `/app/data` y cargá las env vars.
- **Tu propia VPS**: `docker build -t mundial-bot . && docker run -d --env-file .env -v $PWD/data:/app/data mundial-bot`

## Costo
Railway: free trial + ~US$5/mes el plan Hobby. Un solo servicio liviano alcanza.
