-- Capture-session upload tracking: one row per browser-uploaded session.
-- The dashboard inserts (status=uploaded) after the R2 upload completes;
-- the workstation processor (tools/process_incoming.ts) advances the row
-- through processing -> ready | failed and attaches the validation/bake
-- report. Same access posture as sail_audits: the password gate hides the
-- app; anon writes are size-guarded pending the post-sprint auth flip.
create table if not exists public.sail_ingests (
  id         uuid primary key default gen_random_uuid(),
  session    text not null unique,
  status     text not null default 'uploaded'
             check (status in ('uploaded','processing','ready','failed')),
  manifest   jsonb not null default '{}'::jsonb,
  report     jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.sail_ingests enable row level security;

drop policy if exists "sail_ingests read" on public.sail_ingests;
create policy "sail_ingests read" on public.sail_ingests
  for select to anon, authenticated using (true);

drop policy if exists "sail_ingests insert" on public.sail_ingests;
create policy "sail_ingests insert" on public.sail_ingests
  for insert to anon, authenticated
  with check (pg_column_size(manifest) < 262144 and status = 'uploaded');

drop policy if exists "sail_ingests update" on public.sail_ingests;
create policy "sail_ingests update" on public.sail_ingests
  for update to anon, authenticated
  using (true)
  with check (pg_column_size(report) < 262144);

revoke delete on public.sail_ingests from anon, authenticated;
