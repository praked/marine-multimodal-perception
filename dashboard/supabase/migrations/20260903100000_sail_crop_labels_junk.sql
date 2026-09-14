-- Fourth crop verdict: 'junk' = the detection itself is a bad box
-- (loose/whole-frame/not a boat) — distinct from 'deny' (a real boat that
-- is not ours). Also yields detector-precision GT for free. The original
-- check constraint listed only accept/deny/skip, so widen it.
alter table public.sail_crop_labels
  drop constraint if exists sail_crop_labels_label_check;
alter table public.sail_crop_labels
  add constraint sail_crop_labels_label_check
  check (label in ('accept', 'deny', 'skip', 'junk'));
