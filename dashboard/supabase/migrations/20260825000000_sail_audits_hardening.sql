-- sail_audits hardening (2026-08-25).
--
-- The anon insert policy STAYS: the web annotate flow is credential-free and
-- the audit sprint depends on it. This migration only tightens WHAT anon can
-- insert — shape, not access. The access flip lives at the bottom of this
-- file as a commented template, to be applied AFTER the audit sprint and
-- BEFORE any wide sharing of the dashboard.

-- ----------------------------------------------------- record shape guard
-- The web flow writes AuditRecord (dashboard/lib/types.ts AuditRecordSchema)
-- into the jsonb `record` column. Anything anon can insert should at least
-- look like one: a JSON OBJECT carrying the schema's top-level keys, with a
-- verdict from the closed set and `boxes` an array. Readers already
-- safeParse and drop malformed rows, so this guard costs no reader
-- behaviour — it just stops the table becoming an anonymous jsonb dump.
--
-- NOT VALID: applies to new writes only; existing rows (all written by the
-- same web flow) are not scanned, so applying this cannot fail mid-sprint.
-- Validation of the backlog is part of the commented follow-up below.
alter table public.sail_audits
  drop constraint if exists sail_audits_record_shape;
alter table public.sail_audits
  add constraint sail_audits_record_shape check (
    jsonb_typeof(record) = 'object'
    and record ?& array['id', 'clip_key', 'frame_ts', 'frame_id',
                        'verdict', 'boxes', 'source', 'created_at']
    and record->>'verdict' in ('accept', 'reject', 'edit')
    and jsonb_typeof(record->'boxes') = 'array'
    -- the promoted columns must agree with the payload, so a reader
    -- filtering by clip_key sees the same clip the record claims:
    and lower(record->>'id') = lower(id::text)
    and record->>'clip_key' = clip_key
    and record->>'frame_ts' = frame_ts
  ) not valid;

-- -------------------------------------------------- rate-limit monitoring
-- Append-only + anon means abuse shows up as insert VOLUME. A plain
-- created_at index makes "how many rows landed in the last N minutes" (and
-- any future trigger/edge-function rate limiter keyed on recency) an index
-- scan instead of a table scan. sail_audits_clip_idx (clip_key, created_at)
-- cannot serve that query without a leading clip_key.
create index if not exists sail_audits_created_idx
  on public.sail_audits (created_at);

-- ======================================================================
-- FUTURE AUTH MIGRATION — TEMPLATE, DO NOT APPLY YET.
-- When: after the audit sprint completes, before the dashboard is shared
-- beyond the research team. Applying it earlier breaks the credential-free
-- annotate flow (every insert from the web app would be rejected).
-- Copy into a new dated migration file rather than uncommenting here, so
-- the migration history stays append-only.
-- ======================================================================
--
-- -- 1. Validate the backlog now that the sprint's rows are in (fails if
-- --    any historical row violates the shape guard; fix or delete those
-- --    rows with the service role first).
-- alter table public.sail_audits
--   validate constraint sail_audits_record_shape;
--
-- -- 2. Inserts become authenticated-only. Reads stay public (the
-- --    password gate hides the app; the catalogue is already public-read).
-- drop policy if exists "sail_audits append" on public.sail_audits;
-- create policy "sail_audits append"
--   on public.sail_audits for insert
--   to authenticated
--   with check (pg_column_size(record) < 65536);
-- revoke insert on public.sail_audits from anon;
--
-- -- 3. Same treatment for the goals editor (documented here so the flip
-- --    is not done by halves; sail_goals carries the same anon-write
-- --    rationale as sail_audits).
-- drop policy if exists "sail_goals write" on public.sail_goals;
-- create policy "sail_goals write"
--   on public.sail_goals for insert
--   to authenticated
--   with check (pg_column_size(goals) < 32768);
-- drop policy if exists "sail_goals update" on public.sail_goals;
-- create policy "sail_goals update"
--   on public.sail_goals for update
--   to authenticated
--   using (true)
--   with check (pg_column_size(goals) < 32768);
-- revoke insert, update on public.sail_goals from anon;
--
-- -- 4. Prerequisite on the app side (do BEFORE applying 2-3): a login
-- --    flow in dashboard/ (supabase.auth) so annotators hold a session;
-- --    dashboard/lib/audit/backend.ts createSupabaseBackend() then
-- --    inserts as `authenticated` unchanged.
