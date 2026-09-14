-- Swarm self-recognition crop labels (plan docs/plans/swarm_self_recognition.md,
-- decision D2): "is this OUR ASVProject boat?" verdicts on boat-detection crops,
-- labelled on /crops. Append-only like sail_audits: rows are never updated or
-- deleted; readers fold to latest-per-crop. The crop_id is the sha1 the
-- extraction manifest carries (dashboard/tools/swarm_crops/extract_crops.py).
create table if not exists public.sail_crop_labels (
  id uuid primary key default gen_random_uuid(),
  crop_id text not null,
  frame_id text,
  label text not null check (label in ('accept', 'deny', 'skip')),
  created_at timestamptz not null default now()
);
create index if not exists sail_crop_labels_crop_idx
  on public.sail_crop_labels (crop_id, created_at);
alter table public.sail_crop_labels enable row level security;
-- Same posture as sail_audits/sail_plans: the password gate hides the app;
-- anyone holding the anon key could write. Tighten with the audits auth flip.
drop policy if exists sail_crop_labels_all on public.sail_crop_labels;
create policy sail_crop_labels_all on public.sail_crop_labels
  for all to anon, authenticated using (true) with check (true);
