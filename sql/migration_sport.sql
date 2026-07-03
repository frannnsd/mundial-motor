-- Migración multi-deporte (M4): agrega la dimensión sport a las tablas compartidas.
-- El orquestador la ejecuta A MANO en el SQL Editor de Supabase ANTES del deploy.
-- Compatibilidad WC: default 'wc' en ambas tablas → las filas existentes y los
-- callers del Mundial (que no mandan sport) siguen funcionando idéntico.

alter table daily_reports add column if not exists sport text not null default 'wc';
alter table daily_reports drop constraint daily_reports_pkey;
alter table daily_reports add primary key (sport, fixture_id);
alter table props_log add column if not exists sport text not null default 'wc';
create unique index if not exists props_log_sport_key on props_log (sport, fixture_id, player_id, market);
-- la unique vieja (fixture_id,player_id,market) se mantiene: los mercados MLB usan
-- nombres disjuntos (prefijo mlb_/ks) así que nunca colisiona; se borra post-torneo.
