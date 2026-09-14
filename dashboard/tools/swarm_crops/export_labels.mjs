/* Export the swarm-crop labels: manifest joined with the latest label per
 * crop, written as repo-side JSONL for the D.1 crop-classifier training.
 *
 *   node --env-file=.env.local tools/swarm_crops/export_labels.mjs \
 *     --manifest <scratch>/swarm_crops/2026-08-26/manifest.jsonl \
 *     [--out ../labels/swarm_crops_2026-08-26.jsonl] [--all]
 *
 * By default only labelled crops are written (label: accept|deny|skip|junk +
 * labeled_at); --all writes every manifest row (unlabelled get label: null).
 * Reads sail_crop_labels with the service role, paged past the 1000-row cap.
 */

import { createClient } from "@supabase/supabase-js";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const MANIFEST = opt("manifest");
const OUT = opt("out", "../labels/swarm_crops_2026-08-26.jsonl");
const PREDICTIONS = opt(
  "predictions",
  "/Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/predictions.json",
);
const ALL = args.includes("--all");
if (!MANIFEST) {
  console.error("--manifest <manifest.jsonl> is required");
  process.exit(1);
}
if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
  console.error("Supabase creds missing — run with node --env-file=.env.local");
  process.exit(1);
}

const sb = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { persistSession: false } },
);

// latest label per crop_id (append-only log)
const latest = new Map();
const page = 1000;
for (let from = 0; ; from += page) {
  const { data, error } = await sb
    .from("sail_crop_labels")
    .select("crop_id, label, created_at")
    .order("created_at", { ascending: true })
    .range(from, from + page - 1);
  if (error) throw new Error(error.message);
  for (const r of data ?? []) {
    const prev = latest.get(r.crop_id);
    if (!prev || r.created_at >= prev.created_at) latest.set(r.crop_id, r);
  }
  if (!data || data.length < page) break;
}
console.log(`${latest.size} labelled crops in sail_crop_labels`);

// model predictions are joined when present (additive columns)
const preds = existsSync(PREDICTIONS)
  ? JSON.parse(await readFile(PREDICTIONS, "utf8"))
  : null;
if (preds) console.log(`joining model predictions (${Object.keys(preds).length} crops)`);

const rows = (await readFile(MANIFEST, "utf8"))
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

const out = [];
const counts = { accept: 0, deny: 0, skip: 0, junk: 0, unlabelled: 0 };
for (const r of rows) {
  const lab = latest.get(r.crop_id);
  if (lab) counts[lab.label] = (counts[lab.label] ?? 0) + 1;
  else counts.unlabelled++;
  if (!lab && !ALL) continue;
  const pr = preds?.[r.crop_id];
  out.push(
    JSON.stringify({
      ...r,
      label: lab?.label ?? null,
      labeled_at: lab?.created_at ?? null,
      ...(pr ? { predicted_label: pr.label, predicted_p: pr.p } : {}),
    }),
  );
}
await writeFile(OUT, out.join("\n") + (out.length ? "\n" : ""));
console.log(
  `wrote ${out.length} rows -> ${OUT}\n` +
    `accept ${counts.accept} · deny ${counts.deny} · skip ${counts.skip} · junk ${counts.junk} · unlabelled ${counts.unlabelled}`,
);
