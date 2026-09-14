-- ASVProject dashboard core schema.
-- Postgres holds catalogue metadata only; per-frame assets (JPEG frames,
-- masks, sectors/radar/boxes JSON) live as bundle files in the public
-- Storage bucket `asvproject-clips`, laid out exactly like /public/demo
-- (writer: dashboard/tools/bake_demo.py, uploader: dashboard/tools/ingest).

-- ---------------------------------------------------------------- catalogue
create table if not exists public.sail_clips (
  clip_key   text primary key,          -- "<scene>__<triplet_ts>"
  clip_id    text not null unique,      -- "<scene>/<triplet_ts>"
  summary    jsonb not null,            -- ClipSummary (catalogue card)
  meta       jsonb not null,            -- ClipMeta (frame index etc.)
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.sail_clips enable row level security;

-- Read-only catalogue for everyone; writes go through the service role
-- (the ingest CLI), which bypasses RLS.
drop policy if exists "sail_clips public read" on public.sail_clips;
create policy "sail_clips public read"
  on public.sail_clips for select
  to anon, authenticated
  using (true);

revoke insert, update, delete on public.sail_clips from anon, authenticated;

-- ------------------------------------------------------------ audit log
-- Append-only, mirroring the repo's JSONL discipline: no update/delete
-- policies exist, so even authenticated clients can only ever add records.
create table if not exists public.sail_audits (
  id         uuid primary key,
  clip_key   text not null references public.sail_clips (clip_key)
             on delete cascade,
  frame_ts   text not null,
  record     jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists sail_audits_clip_idx
  on public.sail_audits (clip_key, created_at);

alter table public.sail_audits enable row level security;

drop policy if exists "sail_audits read" on public.sail_audits;
create policy "sail_audits read"
  on public.sail_audits for select
  to anon, authenticated
  using (true);

-- NOTE: anon insert keeps the annotation loop credential-free for the
-- research team. Before any properly public deployment, tighten this to
-- `to authenticated` and add a login flow. The size guard stops abuse of
-- the jsonb column.
drop policy if exists "sail_audits append" on public.sail_audits;
create policy "sail_audits append"
  on public.sail_audits for insert
  to anon, authenticated
  with check (pg_column_size(record) < 65536);

revoke update, delete on public.sail_audits from anon, authenticated;

-- ------------------------------------------------------------- storage
-- The public bucket `asvproject-clips` is created by the ingest CLI via the
-- Storage API (public buckets need no storage.objects policy, and newer
-- Supabase projects reject storage DDL from SQL owned by other roles).
-- Uploads happen with the service key only.
