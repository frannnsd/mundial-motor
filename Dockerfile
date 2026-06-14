# Imagen de producción del Mundial Value Bot (deploy cloud 24/7).
# Python 3.12-slim: estable y con wheels para todo el stack científico.
FROM python:3.12-slim

WORKDIR /app

# build-essential: Cython compila penaltyblog.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[model,xg]"

# La banca/picks viven en /app/data → montá un volumen para que persistan.
VOLUME ["/app/data"]

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

# Servicio 24/7: bot conversable + scheduler (diario + pre-partido) en un proceso.
CMD ["python", "scripts/run_bot.py"]
