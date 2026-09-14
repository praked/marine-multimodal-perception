-- Team-level coverage goals for the data map's edit mode: one jsonb row,
-- shared by the team. Same access rationale as sail_audits: the password
-- gate hides the app; anon write is size-guarded and to be tightened when
-- real auth lands.
create table if not exists public.sail_goals (
  id         text primary key default 'default',
  goals      jsonb not null,
  updated_at timestamptz not null default now()
);

alter table public.sail_goals enable row level security;

drop policy if exists "sail_goals read" on public.sail_goals;
create policy "sail_goals read"
  on public.sail_goals for select
  to anon, authenticated
  using (true);

drop policy if exists "sail_goals write" on public.sail_goals;
create policy "sail_goals write"
  on public.sail_goals for insert
  to anon, authenticated
  with check (pg_column_size(goals) < 32768);

drop policy if exists "sail_goals update" on public.sail_goals;
create policy "sail_goals update"
  on public.sail_goals for update
  to anon, authenticated
  using (true)
  with check (pg_column_size(goals) < 32768);

revoke delete on public.sail_goals from anon, authenticated;
