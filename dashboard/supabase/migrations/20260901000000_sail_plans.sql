-- Shared annotation plans: the planner's saved plans move from per-browser
-- localStorage to a shared table so plans built anywhere (including headless
-- sessions) appear for every auditor. Progress cursors stay local per browser.
create table if not exists public.sail_plans (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  requested int not null,
  frames jsonb not null,          -- [{"clipKey","ts"}]
  stats jsonb,
  clamped boolean not null default false,
  created_at timestamptz not null default now()
);
alter table public.sail_plans enable row level security;
-- Same posture as sail_audits: the password gate hides the app; anyone holding
-- the anon key could write. Tighten to authenticated with the audits policy.
drop policy if exists sail_plans_all on public.sail_plans;
create policy sail_plans_all on public.sail_plans
  for all to anon, authenticated using (true) with check (true);
