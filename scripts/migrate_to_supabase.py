"""Migración one-shot del histórico local a Supabase (forward-test + datos).

Sube:
  data/forward_test.sqlite (props_log)            → props_log   (insert_ignore)
  data/nt_cache/nt_matches.csv                    → nt_matches  (upsert por match_id)
  data/players_cache/wc_player_matches.csv        → player_matches (upsert por id)

Idempotente: props_log respeta la semántica inmutable (la primera predicción
gana, on_conflict=ignore); las tablas de datos se upsertean. Al final imprime
conteos ANTES (local) y DESPUÉS (Supabase, count exacto vía PostgREST) y
verifica que la nube tenga al menos todas las filas locales.

Uso:  python scripts/migrate_to_supabase.py
Requiere SUPABASE_URL y SUPABASE_SERVICE_KEY en el entorno o el .env.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Permite ejecutar el script sin instalar el paquete.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundial_bot.wc import store  # noqa: E402

FORWARD_TEST_DB = ROOT / "data" / "forward_test.sqlite"
NT_MATCHES_CSV = ROOT / "data" / "nt_cache" / "nt_matches.csv"
PLAYER_MATCHES_CSV = ROOT / "data" / "players_cache" / "wc_player_matches.csv"
BATCH = 500

CHECKLIST = """Faltan SUPABASE_URL / SUPABASE_SERVICE_KEY. Checklist:
  1. Crear el proyecto en supabase.com (plan free alcanza).
  2. Pegar sql/schema.sql UNA vez en el SQL Editor de Supabase.
  3. Exportar SUPABASE_URL (https://xxxx.supabase.co) y SUPABASE_SERVICE_KEY
     (la service_role key, SOLO en el backend) en el entorno o el .env.
  4. Volver a correr: python scripts/migrate_to_supabase.py"""


def _batches(rows: list[dict], size: int = BATCH):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _remote_count(table: str) -> int:
    """Count exacto vía PostgREST (Prefer: count=exact + Content-Range)."""
    r = requests.get(
        store._url(table),  # noqa: SLF001 — mismo cliente, script interno
        headers=store._headers(prefer="count=exact"),  # noqa: SLF001
        params={"select": "*", "limit": "1"},
        timeout=30,
    )
    r.raise_for_status()
    content_range = r.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError(f"Sin Content-Range al contar {table}: {content_range!r}")
    return int(content_range.rsplit("/", 1)[1])


def _local_props_rows() -> list[dict]:
    """Filas del props_log local, sin el id autoincremental (Postgres genera el suyo)."""
    if not FORWARD_TEST_DB.exists():
        return []
    with sqlite3.connect(FORWARD_TEST_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM props_log ORDER BY id").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d.pop("id", None)
        out.append(d)
    return out


def _csv_records(path: Path) -> list[dict]:
    """CSV → records JSON-safe (NaN → null; fechas quedan como texto tal cual)."""
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8")
    return json.loads(df.to_json(orient="records"))


def _push(table: str, rows: list[dict], *, on_conflict: str, mode: str) -> None:
    print(f"  Subiendo {len(rows)} filas a {table} en lotes de {BATCH}...")
    inserted = 0
    for batch in _batches(rows):
        if mode == "insert_ignore":
            inserted += store.insert_ignore(table, batch, on_conflict=on_conflict)
        else:
            store.upsert(table, batch, on_conflict=on_conflict)
            inserted += len(batch)
    verb = "insertadas" if mode == "insert_ignore" else "upserteadas"
    print(f"  {table}: {inserted} filas {verb}.")


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not store.is_configured():
        print(CHECKLIST)
        return 1

    props = _local_props_rows()
    nt_csv = _csv_records(NT_MATCHES_CSV)
    pm_csv = _csv_records(PLAYER_MATCHES_CSV)
    nt_rows = [{"match_id": str(r["match_id"]), "payload": r} for r in nt_csv]
    pm_rows = [{"id": f"{int(r['fixture_id'])}_{int(r['player_id'])}", "payload": r}
               for r in pm_csv]

    plan = [
        ("props_log", props, "fixture_id,player_id,market", "insert_ignore"),
        ("nt_matches", nt_rows, "match_id", "upsert"),
        ("player_matches", pm_rows, "id", "upsert"),
    ]

    print("Conteos LOCALES (antes):")
    for table, rows, _oc, _mode in plan:
        print(f"  {table}: {len(rows)}")

    for table, rows, on_conflict, mode in plan:
        if not rows:
            print(f"  {table}: sin datos locales, salteo.")
            continue
        _push(table, rows, on_conflict=on_conflict, mode=mode)

    print("\nConteos en SUPABASE (después):")
    ok = True
    for table, rows, _oc, _mode in plan:
        remote = _remote_count(table)
        status = "OK" if remote >= len(rows) else "FALTAN FILAS"
        if remote < len(rows):
            ok = False
        print(f"  {table}: local {len(rows)} → remoto {remote}  [{status}]")

    if not ok:
        print("\nERROR: la nube tiene menos filas que lo local. Revisar y re-correr "
              "(la migración es idempotente).")
        return 1
    print("\nMigración verificada: la nube contiene todo el histórico local.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
