-- Set curation (2026-08-28): soft-delete + trim ranges per clip key.
--
-- Metadata, never file operations. A row here says what the corpus SHOULD
-- show; the R2 bundle and the raw captures on the SSD are untouched by any
-- write to this table. The only thing that removes bytes is the explicit
-- `pnpm curation:prune --yes` (service role), which purges the R2 bundle +
-- catalogue row of sets deleted > 30 days ago and stamps `purged_at`.
--
-- Keys are dashboard clip keys "<scene>__<triplet_ts>". A key may name a
-- concatenated ACTIVITY (the catalogue rows in sail_clips) or a single
-- capture CHUNK that never had a bundle (the baker's former hard-coded
-- exclusion list, seeded below) — so there is deliberately no FK to
-- sail_clips.
--
-- Deleted   := deleted_at is not null and (restored_at is null or
--              restored_at < deleted_at), or purged_at is not null.
-- Cuts      := jsonb array of {start_ts, end_ts, note} in frame-id timestamp
--              units ("HH:MM:SS.f", inclusive both ends). Frame ids are
--              unchanged by a cut, so every label / audit / sector record
--              still lines up; consumers just skip those frames.
--
-- Access posture = sail_audits: the password gate hides the app; anon
-- writes are shape/size-guarded pending the post-sprint auth flip
-- (20260825000000_sail_audits_hardening.sql, template at the bottom).

create table if not exists public.sail_curation (
  clip_key    text primary key,
  deleted_at  timestamptz,
  restored_at timestamptz,
  purged_at   timestamptz,          -- set ONLY by the prune tool (service role)
  cuts        jsonb not null default '[]'::jsonb,
  note        text,
  updated_at  timestamptz not null default now(),
  updated_by  text                  -- free text: who / which tool
);

alter table public.sail_curation
  drop constraint if exists sail_curation_cuts_shape;
alter table public.sail_curation
  add constraint sail_curation_cuts_shape check (
    jsonb_typeof(cuts) = 'array' and pg_column_size(cuts) < 65536
  );

alter table public.sail_curation enable row level security;

drop policy if exists "sail_curation read" on public.sail_curation;
create policy "sail_curation read" on public.sail_curation
  for select to anon, authenticated using (true);

-- The web app upserts (insert-or-update). A purged row is frozen for the
-- web: its bundle is gone, so a "restore" would point at nothing. Only the
-- service role (prune) may write purged_at.
drop policy if exists "sail_curation insert" on public.sail_curation;
create policy "sail_curation insert" on public.sail_curation
  for insert to anon, authenticated
  with check (purged_at is null);

drop policy if exists "sail_curation update" on public.sail_curation;
create policy "sail_curation update" on public.sail_curation
  for update to anon, authenticated
  using (purged_at is null)
  with check (purged_at is null);

revoke delete on public.sail_curation from anon, authenticated;

-- ------------------------------------------------------------ history
-- Append-only, like sail_audits: every delete / restore / cut edit / purge
-- lands here with the cuts as they stood after the action.
create table if not exists public.sail_curation_log (
  id         uuid primary key default gen_random_uuid(),
  clip_key   text not null,
  action     text not null
             check (action in ('delete', 'restore', 'set_cuts', 'purge', 'seed')),
  cuts       jsonb,
  note       text,
  created_at timestamptz not null default now(),
  created_by text
);

create index if not exists sail_curation_log_clip_idx
  on public.sail_curation_log (clip_key, created_at);

alter table public.sail_curation_log enable row level security;

drop policy if exists "sail_curation_log read" on public.sail_curation_log;
create policy "sail_curation_log read" on public.sail_curation_log
  for select to anon, authenticated using (true);

drop policy if exists "sail_curation_log append" on public.sail_curation_log;
create policy "sail_curation_log append" on public.sail_curation_log
  for insert to anon, authenticated
  with check (action <> 'purge' and pg_column_size(cuts) < 65536);

revoke update, delete on public.sail_curation_log from anon, authenticated;

-- ------------------------------------------------------- seed (one-off)
-- The baker's hard-coded EXCLUDE_CHUNK_IDS (chunk-level first/last-frame
-- screening, AuthorTwo 2026-08-21) become curation rows so nothing that was
-- excluded before this table existed reappears once the list is gone from
-- the code. These chunks never had a bundle: nothing to prune, nothing to
-- restore from the UI (they are not catalogue rows); the exported
-- configs/curation.yaml carries them to the Python side.
insert into public.sail_curation (clip_key, deleted_at, note, updated_by)
values
  ('2026-06-17_institutionone_day1__2026-06-17_12-31-00', '2026-08-21T00:00:00Z', 'indoor office', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-06-17_institutionone_day1__2026-06-17_12-31-23', '2026-08-21T00:00:00Z', 'dock deck setup', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-03-11', '2026-08-21T00:00:00Z', 'pier setup', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-04-11', '2026-08-21T00:00:00Z', 'indoor laptop closeup', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-08-12', '2026-08-21T00:00:00Z', 'pier walkway', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-13-12', '2026-08-21T00:00:00Z', 'pier walkway', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-18-12', '2026-08-21T00:00:00Z', 'pier + person', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-39-56', '2026-08-21T00:00:00Z', 'pontoon deck teardown', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-44-56', '2026-08-21T00:00:00Z', 'pontoon deck teardown', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_16-50-03', '2026-08-21T00:00:00Z', 'indoor office', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-07-08__2026-07-08_18-26-45', '2026-08-21T00:00:00Z', 'indoor office', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-08-18_pontoon__2026-08-18_19-13-32', '2026-08-21T00:00:00Z', 'indoor lab desk', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-08-18_pontoon__2026-08-18_19-18-32', '2026-08-21T00:00:00Z', 'indoor lab desk', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-08-18_pontoon__2026-08-18_19-23-32', '2026-08-21T00:00:00Z', 'indoor lab desk', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-08-18_pontoon__2026-08-18_19-28-33', '2026-08-21T00:00:00Z', 'indoor lab desk', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'),
  ('2026-08-19_afloat_session__2026-08-19_18-43-07', '2026-08-21T00:00:00Z', 'indoor changing room', 'migration:bake_corpus.EXCLUDE_CHUNK_IDS')
on conflict (clip_key) do nothing;

insert into public.sail_curation_log (clip_key, action, note, created_by)
select clip_key, 'seed', note, updated_by
from public.sail_curation
where updated_by = 'migration:bake_corpus.EXCLUDE_CHUNK_IDS'
  and not exists (
    select 1 from public.sail_curation_log l
    where l.clip_key = sail_curation.clip_key and l.action = 'seed'
  );

-- ------------------------------------------------ audits survive a purge
-- sail_audits referenced sail_clips with ON DELETE CASCADE (core schema).
-- With retention pruning, deleting a catalogue row would silently destroy
-- the set's audit log — human work, not derived data. Audits are keyed by
-- clip_key + frame_id text and stay meaningful without the catalogue row
-- (the Python side matches them by frame id), so the FK goes. Readers
-- already tolerate audits for clips they cannot find.
do $$
declare c record;
begin
  for c in
    select conname from pg_constraint
    where conrelid = 'public.sail_audits'::regclass and contype = 'f'
  loop
    execute format('alter table public.sail_audits drop constraint %I', c.conname);
  end loop;
end $$;
