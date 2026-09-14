# ASVProject Dashboard

Web recreation of the obstacle-detection data visualisation + annotation
dashboard: Next.js on Vercel, Supabase for data, no local datasets required.
The visual identity follows the Institution One corporate design —
Seeblau (Pantone 2995) as the single accent over Papierweiß, with a full dark
mode restating the same tokens.

## What it does

- **Clips** (`/clips`) — catalogue of mission clips with stream chips
  (fisheye / thermal / radar / IMU / seg / sectors; absent streams are shown
  struck-through, because absence is information). Each card has a **data
  link** button (and a multi-select checkbox + floating bar for a combined
  link): `/api/export` mints a 7-day presigned fetch script —
  `curl -fsSL '<url>' | bash` pulls the selected bundles onto any machine
  (a GPU box) with no dashboard login there. An amber toast covers the
  mint (1–2 min for large clips: one presigned GET per object), green/red
  reports the outcome. Header links to `/upload`.
- **Viewer** (`/clips/<scene>__<ts>`) — the four-panel pipeline view,
  1:1 with `images/pipeline_system_in_action.png`:
  1. *Fisheye* — undistorted frame, eWaSR water/sky/obstacle tint, amber
     water edge, typed detection boxes, instance masks, label boxes; IMU
     attitude readout in the panel header. Optional **radar → fisheye
     projection** layer (`lib/viewer/radarProjection.ts`, the TS twin of
     `geometry.project_radar_to_undistorted` + the pipeline's −6.0° mount
     yaw, fixture-tested against the Python): returns drawn at their
     projected pixels, size falls with range, colour by Doppler
     (approach red / recede blue / static radar-purple); respects the
     clutter-filter toggle.
  2. *Thermal* — honest 160×120 pixels (or an explicit DOWN state).
  3. *Radar bird's-eye* — ±55° wedge grid, 3/6/9 m range arcs, fading trail,
     optional `y ≥ 0.5 m` mount-clutter filter; protocol-v1.2 **tracked
     targets** (optional `targets` field, "Target tracks" toggle): one
     motion-state-coloured ring per track (closing = serious red,
     crossing = amber, static/diverging muted) with a speed-proportional
     velocity arrow, a dashed CPA ghost (position + v·t_cpa) when the CPA
     is predicted, and a mono chip row (state · range · closing · CPA)
     under the panel.
  4. *Sector polar* — per-bin wedges (fused score, learned-scorer
     p(obstacle), or threat), radar-confirmed ▾ markers, p=0.5/1.0 arcs.
  Plus the per-bin fusion table (follows the sector-value toggle), heading
  cost curve + recommendation, transport bar (space, ←/→, shift+←/→),
  persisted layer toggles, and a **sector-width slider in true degrees**
  (5–45°, 5° steps, default 15° = the D.2 native bin): `lib/sectors.ts`
  resamples the record's native bins — 10° legacy or 15°, read per record
  from `bin_centers_deg` — onto a dead-ahead-centered grid, taking the
  conservative aggregate (max score, min range, any confirmed) of every
  overlapped bin; sub-native widths subdivide without inventing resolution.
  **Learned p(obstacle) is the default sector value** (calibrated scorer;
  fused n/3 one click away; automatic fallback for clips without scorer
  fields). Per-sensor **health timelines are tri-state**: green ok /
  pastel-amber degraded (thermal below the quality-guard contrast floors,
  `thermal_q:"low"` stamped at bake) / red down.
- **Coverage map** (`/datamap`) — the data-completion map (portable
  `components/datamap/`): sky-dome sun paths, coverage rings (24 h clock,
  cloud/wind/precipitation/wind-direction), luminance spectrum, entity
  gauges — every sector with a hover provenance bubble; **editable goals**
  (shared `sail_goals` row, localStorage in demo) drive the gap analysis
  and the gaps-dropdown KPI; **advanced mode** adds the entity × condition
  intersection matrix (amber ring = deployment-risk gap: class exists in
  the corpus but never captured under that condition).
- **3D replay** (`/replay`) — LGL-BW orthophoto ground, NASA sun + moon
  models on real ephemeris (`lib/celestial.ts`); boat model + IMU yaw
  wiring pending.
- **Annotate** (`/annotate`) — the audit workflow, two layers:
  - **Session planner** (`/annotate`) — deterministic best-N frame
    selection across the corpus (`lib/annotate/planner.ts`, the TS port of
    `scripts.eval.audit_frame_selector`): scorer/label disagreement,
    rarity-weighted class quotas, condition-cell diminishing returns,
    ≥8-frame spacing; asking for more than exists returns the maximum with
    a notice. **Advanced conditions**: stackable NOT/AND/OR/XOR rows over
    instances and capture conditions (left-to-right fold), every control
    with a hover tooltip explaining its semantics. Built plans **persist
    in the browser** (frames + cursor + coverage stats): revisiting shows
    "Saved plan · at frame k/N" with continue/discard.
  - **Frame audit** (`/annotate/<clip>`, `?plan=1` for plan mode) —
    keyboard-first: `a` accept, `r` reject, `d` draw, `x` delete, `u`/⌘Z
    undo, ⇧⌘Z redo, `c` place a pseudo-centroid inside the selected box,
    `s` structure, space play, `b D B p m o` classes, `[`/`]` or ←/→ walk
    the plan in plan mode (⌥←/→ steps clip frames); the header's
    plan/audited chips toggle between plan and clip modes; boxes select by
    box or label click. The log is **append-only** (localStorage in demo
    mode, `sail_audits` in Supabase mode) and exports the repo's
    training-frames JSONL shape (plan-aware: gathers all plan clips, named
    after the plan; centroids ride along as an optional `centroid` field).
    Teacher chips (all/DINO/DART/agreed) filter the suggestion seed of
    unaudited frames; `masks` toggles SAM3 polygon overlays; `reseed`
    replaces the editor's boxes with the teacher suggestions (recovers a
    frame whose audit buried them — the next saved verdict supersedes).
    Hardened under the tranche-A audit (2026-09-02): labels.json is served
    gzip'd (Content-Encoding; ingest uploads it compressed), the loader
    retries each network piece 3× and treats the offline pack strictly as
    a fallback (red "stale pack — reload" chip when online), packs
    auto-heal after every successful online load, and audit writes retry
    then QUEUE on any persistent failure (pending-sync pill) instead of
    being lost. Server-side export of the whole audit log:
    `node tools/export_audits.mjs` → `labels/audited/`. Minimum drawn box
    is 2 native px (distant ducks/buoys are genuinely that small).
- **Upload** (`/upload`, linked from `/clips`) — browser upload of a
  capture session folder: chunk-pairing validation first (fisheye+mmwave
  required per chunk, thermal honest-absent, non-capture files refused),
  then direct-to-bucket presigned PUTs (whitelisted names, 3 GB/file cap,
  content-length pinned into the signature) into `incoming/<session>/`,
  then a `sail_ingests` tracking row. The page polls the row so
  uploaded → processing → ready/failed is visible. The processing itself
  runs on a workstation: `pnpm process:incoming` (`--watch` to poll) pulls
  queued sessions from R2 onto the SSD and runs the ingest_session chain
  (validate → bake `--only` → enrich → `ingest_r2 --prune`), reporting
  status + errors back into the row. Refuses to run without the SSD.
- **Curation — delete + trim sets** (`/clips` + the viewer). Metadata,
  never file operations: `public.sail_curation` holds one row per clip
  key (soft `deleted_at` / `restored_at`, `purged_at`, `cuts` jsonb), with
  an append-only `sail_curation_log`. On `/clips` a set is deleted with a
  two-click confirm; the **Deleted (N)** tab shows deletion date, days
  until prune and **Restore** (instant; the bundle is kept 30 days). Deleted
  sets vanish from the catalogue, planner, data links (`/api/export` → 410),
  coverage map, replay and offline packs. In the viewer, **in/out points**
  (`i`/`o` or the transport buttons) create **cuts**; cut regions render
  greyed on the scrub bar and the health timelines with draggable edge
  handles, the *Cuts* panel edits timestamps/notes, seeks and removes them,
  and playback + stepping skip cut frames unless *show cut frames* is on.
  Frame ids never change, so labels/audits/sectors keep lining up; the
  clips card shows a *trimmed* chip (kept/cut minutes), the viewer header
  the cut count, and a deleted set shows a "Deleted on … — restore" banner.
  Export to the Python side: `pnpm curation:export` writes the git-tracked
  `configs/curation.yaml` (schema: `docs/reference/data_formats.md` §6) read
  by `scripts/utils/curation.py` (`list_triplets`, `iterate_triplet`, the
  baker — whose hard-coded exclusion list migrated into the table —
  `build_features`, the audit planner). Retention: `pnpm curation:prune`
  lists sets deleted > 30 days ago (dry run); `pnpm curation:prune --yes`
  removes their R2 bundle objects + catalogue row and stamps `purged_at`
  (audit rows kept; curation record kept so the Python side stays
  consistent; raw SSD captures are never touched by anything here).
- **Offline annotation** (`/offline`, PWA). The dashboard registers a
  service worker (`public/sw.js`) that caches the app shell as pages are
  visited and answers page-facing asset URLs from **offline packs**. On
  `/offline` pick the saved annotation plan (from the planner's browser
  storage) or a whole set, see the size estimate (~75 KB/frame + thermal),
  press Download: fisheye + thermal frames go to Cache Storage, meta /
  labels / an audit snapshot to IndexedDB, `navigator.storage.persist()` is
  requested; per-pack status, size, frame count and delete. Packs never
  include deleted sets or cut frames. With the network off, `/annotate/<clip>`
  reads the pack (an *offline pack · N frames* chip; only packed frames are
  shown) and audits go to a local IndexedDB queue — the top strip shows
  **"N audits pending sync"** with a Sync button, and sync also runs
  automatically on reconnect. The log is append-only and every record's
  client uuid is the row's primary key, so a retried sync never duplicates
  (duplicate-key = already synced). The gate cookie is set by `/login`;
  cached pages serve offline because those requests never reach the
  middleware. **Packs live in the browser profile** — they are not isolated
  between different users of the same profile, and an unvisited page is not
  cached (visit `/annotate/<clip>` once online). Live check of both features:
  `node tools/verify/live.mjs` (Playwright, from `dashboard/`).

## Data modes

The provider seam (`lib/data/`) picks a backend at startup:

| Mode | When | Catalogue | Assets | Audits |
|---|---|---|---|---|
| **demo** | no Supabase env | `/public/demo/clips.json` | `/public/demo/...` | localStorage |
| **supabase** | both `NEXT_PUBLIC_SUPABASE_*` set | `public.sail_clips` | Supabase Storage | `public.sail_audits` |
| **supabase + R2** | …plus `NEXT_PUBLIC_ASSET_BASE=/api/assets` and server-side `R2_*` | `public.sail_clips` | **private R2 bucket** via signed redirects | `public.sail_audits` |

Production runs the R2 mode: `/api/assets/<path>` validates the path,
presigns a GET on the private bucket (signing timestamps quantised to the
hour so browsers cache effectively) and 302-redirects. Credentials never
reach the client.

**Access gate**: when `DASHBOARD_PASSWORD` is set, `middleware.ts` requires a
login cookie for every page and API route (including the demo bundle) —
the corpus is unpublished research data. Unset in CI/local dev = gate off.

All modes serve the **same bundle layout**. The committed demo fixture is
written by `tools/bake_demo.py`; the full corpus (10 ACTIVITIES — successive
chunks ≤45 s apart are concatenated, per-frame `chunk` attribution in
meta.json — ~1.5 GB, every-frame half-res from the SSD) by
`tools/bake_corpus.py` (`--force` rebakes when seg/det/labels/sectors
changed under an unchanged chunk set), then **always**
`tools/enrich_bundle.py` (sun/daypart/luminance/weather + per-stream entity
counts — a bake drops enrichment), then `pnpm ingest:r2`. One-command
capture ingest (Pi → SSD md5-verified → validate → bake → enrich → upload):
`tools/ingest_session.py`; the browser-upload variant (`/upload` →
`incoming/<session>/` in the bucket → `pnpm process:incoming` on a
workstation with the SSD) runs the same chain from the download step and
reports through `sail_ingests`. ⚠ ingest skips unchanged files BY SIZE —
after a content-only rebake, md5-spot-check against the bucket:

```
<clipKey>/                      # <scene>__<triplet_ts>
  meta.json                     # ClipMeta + ordered frame index
  frames/ts=<HH-MM-SS.f>.jpg    # undistorted fisheye (432×324)
  thermal/ts=<...>.jpg          # 160×120
  seg/ts=<...>.png              # label-encoded mask (0 obst / 1 water / 2 sky)
  sectors.json                  # {ts: sector-protocol v1.1 (+scorer) record}
  radar.json                    # {ts: [[x,y,z,v], ...]} all radar groups
  boxes.json / instances.json / labels.json
```

Everything is keyed by the repo's RoundedTime `frame_id` convention, so
records round-trip to `labels/*.jsonl` unchanged.

## Local development

```bash
cd dashboard
pnpm install
pnpm dev            # demo mode: uses the committed bundle under public/demo
pnpm test           # vitest: unit + component + bundle-contract tests
pnpm lint && pnpm typecheck && pnpm build
```

Re-bake the demo bundle (needs the repo venv + local data/):

```bash
.venv/bin/python dashboard/tools/bake_demo.py
```

## Deploying

1. **Vercel** — import the repo, set *Root Directory* to `dashboard/`.
   With no env vars the deploy serves the demo bundle.
2. **Supabase** — create a project, run
   `supabase/migrations/20260806000000_sail_core.sql` (SQL editor or
   `supabase db push`), then upload data:
   ```bash
   SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... pnpm ingest [bundleDir]
   ```
3. Set `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY` on the
   Vercel project and redeploy — the app switches to Supabase mode.
4. **Full corpus on private R2**: create a private R2 bucket + an
   Object-Read-&-Write token, fill the `R2_*` vars, then
   `bake_corpus.py` → `pnpm ingest:r2`. On Vercel set the four `R2_*` vars,
   `NEXT_PUBLIC_ASSET_BASE=/api/assets` and `DASHBOARD_PASSWORD`, redeploy.

⚠ The audit-insert policy allows `anon` at the Postgres level; the password
gate hides the whole app, but anyone holding the anon key could still write
audits directly. Tighten the policy to `authenticated` if that matters.

### Auto-deploy from GitHub — CONNECTED 2026-09-07 via a fork mirror

Vercel's GitHub import only lists the linked GitHub user's own namespace and
its organisations, so a collaborator can never import another person's
**private personal** repository, however the Vercel GitHub App is installed
there (AuthorOne installed it on `gh-handle-one` on 2026-09-07; the CLI still answered
"make sure you have access"). The project is therefore connected to a private
fork, `AuthorTwoIsCoding/ASVProject-ObstacleDetection`, Production Branch `main`,
Root Directory `dashboard` (set through the API; the CLI link left it at `.`).
The laptop's `origin` has **two push URLs** (gh-handle-one + the fork), so a single
`git push origin main` updates both and the fork never drifts:

```bash
git remote set-url --add --push origin https://github.com/gh-handle-one/ASVProject-ObstacleDetection.git
git remote set-url --add --push origin https://github.com/AuthorTwoIsCoding/ASVProject-ObstacleDetection.git
```

Anyone else pushing to `gh-handle-one` alone does not deploy; either add the same
second push URL or push the fork too. Long-term fix: move the repo into a
GitHub organisation both are members of, then `vercel git connect` to it and
drop the second push URL. The original notes follow.

### Auto-deploy from GitHub (original notes, 2026-08-25)

Connecting the Vercel project to the GitHub repo (so every push deploys —
`main` to production, other branches to previews) could not be completed
from the CLI: `vercel git connect https://github.com/gh-handle-one/
ASVProject-ObstacleDetection.git` (run 2026-08-25, logged in as
`authortwoiscoding`, project `asvproject-dashboard` linked) fails with *"Failed
to connect gh-handle-one/ASVProject-ObstacleDetection to project … make sure you
have access to the repository"*. The Vercel GitHub App is not authorised
for the `gh-handle-one` account, and granting that is a browser + repo-owner flow
by design. To finish it:

1. Browser: [vercel.com](https://vercel.com) → the team's
   `asvproject-dashboard` project → **Settings → Git → Connect Git
   Repository → GitHub** → pick `gh-handle-one/ASVProject-ObstacleDetection`.
2. If the repo is not listed, the Vercel GitHub App needs installing on
   the `gh-handle-one` account first: <https://github.com/apps/vercel> →
   **Configure** → `gh-handle-one` → grant access to
   `ASVProject-ObstacleDetection`. Only the repo owner (AuthorOne) can approve
   this if your GitHub user lacks admin on the repo.
3. Verify *Root Directory* is `dashboard/` and *Production Branch* is
   `main` in the project's Git settings, then push a trivial commit to
   confirm a deployment appears.

Once the app is authorised, the CLI route also works:
`cd dashboard && vercel git connect` (the project link lives in the
untracked `.vercel/project.json`; re-create with `vercel link` if absent).

## Design system

Tokens live in `app/globals.css` (Tailwind v4 `@theme inline`, CSS variables
per theme). Chart colours were validated with the dataviz six-checks
(CVD separation, lightness bands, contrast):

- Sensor trio — fisheye `#1487B8`, thermal `#B8741A`, radar `#7A6BB5`
  (light); `#1D9BD1` / `#C68018` / `#8F7FD0` (dark).
- Sequential = the Seeblau rasterisation ramp (20/35/65/100 + deep).
- Amber `#F2A33C` is reserved for the water edge + threshold marks (as on
  the poster); status colours are reserved and always carry an icon + label.
- Detection-class colours mirror `scripts/utils/detections.CLASS_COLOURS`
  and the label tool so annotators keep their colour vocabulary.

## Tests & CI

`.github/workflows/dashboard.yml` runs lint, typecheck, vitest and a
no-credentials build on every dashboard change. The vitest suite includes
contract tests that parse the real baked bundle with the zod schemas — if
`bake_demo.py` and `lib/types.ts` drift, CI fails.
