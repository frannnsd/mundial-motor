-- Esquema Supabase del sistema Mundial (pegar UNA vez en el SQL Editor de Supabase).
-- Storage único compartido por el backend (Render) y la web (vía la API del backend).
-- El forward-test es EL activo: predicciones inmutables (primera gana), cuotas se
-- adjuntan a la fila, liquidación solo completa campos vacíos.

-- Predicciones + forward-test + cuotas (espejo del props_log local, en Postgres).
create table if not exists props_log (
    id           bigint generated always as identity primary key,
    created_at   timestamptz not null default now(),
    fixture_id   bigint not null,
    match        text not null,
    market       text not null,
    player_id    bigint not null default 0,   -- 0 = mercado de equipo (team_*)
    player_name  text not null default '-',
    pred_mean    double precision,
    pred_prob    double precision,
    line         double precision,
    odds         double precision,
    book         text,
    stake        double precision,             -- opcional, lo carga el humano
    odds_added_at timestamptz,
    actual       double precision,
    settled_at   timestamptz,
    brier        double precision,
    notes        text default '',
    unique (fixture_id, player_id, market)
);

-- Payload diario por partido (lo que la web muestra: cantidades+pmfs+mercados+props).
create table if not exists daily_reports (
    fixture_id   bigint primary key,
    report_date  date not null,
    kickoff_utc  timestamptz not null,
    home         text not null,
    away         text not null,
    round        text,
    is_knockout  boolean not null default false,
    payload      jsonb not null,               -- cantidades, pmfs, mercados 90/TE, props
    xi_confirmed boolean not null default false,
    deltas       jsonb,                        -- deltas al confirmarse el XI
    updated_at   timestamptz not null default now()
);
create index if not exists daily_reports_date on daily_reports (report_date);

-- Histórico de selecciones (seed por migración; el settle nocturno agrega filas).
create table if not exists nt_matches (
    match_id     text primary key,
    payload      jsonb not null                -- la fila completa del esquema nt_data
);

-- Partidos por jugador del Mundial (ídem).
create table if not exists player_matches (
    id           text primary key,             -- f"{fixture_id}_{player_id}"
    payload      jsonb not null
);

-- Observabilidad de jobs.
create table if not exists job_runs (
    id           bigint generated always as identity primary key,
    job          text not null,
    started_at   timestamptz not null,
    finished_at  timestamptz,
    status       text not null default 'running',  -- running|ok|error
    detail       text default '',
    api_calls    integer default 0
);
create index if not exists job_runs_job on job_runs (job, started_at desc);

-- Backup diario del forward-test (app-level; además la web permite descargar el dump).
create table if not exists backups (
    backup_date  date primary key,
    payload      jsonb not null,
    created_at   timestamptz not null default now()
);
