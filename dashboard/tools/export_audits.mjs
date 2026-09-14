/* Export the audited boxes from sail_audits into repo-format label JSONL
   under labels/audited/det_<scene>.jsonl (one record per frame, LATEST
   audit wins; frame_id/scene/chunk attribution was resolved by the
   dashboard when the audit was written, so no timeline mapping happens
   here). Boxes are human-audited: confidence 1.0, audited: true.
   Usage: node tools/export_audits.mjs  (env from .env.local) */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import pg from "pg";

const env = Object.fromEntries(
  readFileSync(new URL("../.env.local", import.meta.url), "utf8")
    .split("\n").filter((l) => l.includes("=") && !l.startsWith("#"))
    .map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]),
);

const c = new pg.Client({ connectionString: env.SUPABASE_DB_POOLER_URL, ssl: { rejectUnauthorized: false } });
await c.connect();
const rows = (await c.query(`select clip_key, frame_ts, record, created_at from sail_audits order by created_at`)).rows;
await c.end();

const latest = new Map(); // clip_key|frame_ts -> record (latest wins)
for (const r of rows) latest.set(`${r.clip_key}|${r.frame_ts}`, r.record);

const byScene = new Map();
let skipped = 0;
for (const rec of latest.values()) {
  if (!rec.frame_id) { skipped++; continue; }
  const [scene, triplet_ts] = rec.frame_id.split("/");
  const out = {
    frame_id: rec.frame_id,
    scene,
    triplet_ts,
    frame_ts: rec.frame_ts,
    source: "human-audit",
    model_version: "authortwo-tranche-a-2026-09",
    audited: true,
    verdict: rec.verdict,
    // canonical undistorted-fisheye frame size the dashboard serves; the
    // normalised boxes are relative to it (consumers like thermal_gt_eval
    // need pixel dimensions to rebuild bearings)
    width: 864,
    height: 648,
    fisheye_bboxes: rec.boxes.filter((b) => (b.space ?? "fisheye") === "fisheye").map((b) => ({
      cls: b.cls,
      xyxy: b.xyxy,
      confidence: 1.0,
      ...(b.source ? { seed_source: b.source } : {}),
      ...(b.centroid ? { centroid: b.centroid } : {}),
    })),
  };
  // Pure-thermal annotation (2026-09-12): boxes drawn on the Lepton frame
  // itself, normalised to the recorded (upright) 160x120 thermal image.
  // Consumers turn them into bearings with the thermal linear model
  // (cx 77 px, 2.82 px/deg); they carry no monocular range.
  const tb = rec.boxes.filter((b) => b.space === "thermal");
  if (tb.length) {
    out.thermal_width = 160;
    out.thermal_height = 120;
    out.thermal_bboxes = tb.map((b) => ({
      cls: b.cls,
      xyxy: b.xyxy,
      confidence: 1.0,
      ...(b.source ? { seed_source: b.source } : {}),
    }));
  }
  if (!byScene.has(scene)) byScene.set(scene, []);
  byScene.get(scene).push(out);
}

const dir = new URL("../../labels/audited/", import.meta.url).pathname;
mkdirSync(dir, { recursive: true });
let frames = 0, boxes = 0;
for (const [scene, recs] of byScene) {
  recs.sort((a, b) => a.frame_id.localeCompare(b.frame_id));
  writeFileSync(path.join(dir, `det_${scene}.jsonl`), recs.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const nb = recs.reduce((s, r) => s + r.fisheye_bboxes.length, 0);
  frames += recs.length; boxes += nb;
  console.log(`${scene}: ${recs.length} frames, ${nb} boxes`);
}
console.log(`TOTAL: ${frames} frames, ${boxes} boxes${skipped ? ` (${skipped} records without frame_id SKIPPED)` : ""}`);
