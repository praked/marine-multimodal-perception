/* Retention prune: remove the R2 bundle objects + the sail_clips catalogue
 * row of every set that has been DELETED (and not restored) for longer than
 * the retention window, then stamp `purged_at` and log a `purge` row.
 *
 * This is the ONLY code path in the dashboard that removes stored data, and
 * it only ever touches the derived bundle in the bucket + its catalogue row:
 * the raw captures on the SSD are never touched, the curation record stays
 * (so the set stays deleted for the Python side), and the set's audit log
 * (sail_audits) is kept — human work is not derived data.
 *
 * Without --yes it prints what it WOULD purge and exits. Nothing is deleted.
 *
 * Usage (from dashboard/, env from .env.local; needs the service role key):
 *   pnpm curation:prune            # dry run: list candidates
 *   pnpm curation:prune --yes      # purge them
 *   pnpm curation:prune --days 45  # different retention (default 30)
 */

import { DeleteObjectsCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { purgeCandidates } from "../../lib/curation/logic";
import { CurationRecordSchema, RETENTION_DAYS } from "../../lib/curation/types";
import { flag, loadEnvLocal, option, r2, supabaseAdmin } from "./env";

async function listPrefix(s3: ReturnType<typeof r2>["s3"], bucket: string, prefix: string) {
  const keys: { key: string; size: number }[] = [];
  let token: string | undefined;
  do {
    const page = await s3.send(new ListObjectsV2Command({ Bucket: bucket, Prefix: prefix, ContinuationToken: token }));
    for (const o of page.Contents ?? []) if (o.Key) keys.push({ key: o.Key, size: o.Size ?? 0 });
    token = page.NextContinuationToken;
  } while (token);
  return keys;
}

async function main(): Promise<number> {
  loadEnvLocal();
  const yes = flag("--yes");
  const days = Number(option("--days") ?? RETENTION_DAYS);
  if (!Number.isFinite(days) || days < 0) throw new Error("--days must be a non-negative number");
  const supabase = supabaseAdmin({ requireServiceRole: true });
  const { s3, bucket } = r2();

  const { data: rows, error } = await supabase.from("sail_curation").select("*");
  if (error) throw new Error(`sail_curation select: ${error.message}`);
  const records = (rows ?? [])
    .map((r) => CurationRecordSchema.safeParse(r))
    .filter((p) => p.success)
    .map((p) => p.data);
  const now = Date.now();
  const candidates = purgeCandidates(records, now, days);
  console.log(`${records.length} curation record(s); ${candidates.length} deleted > ${days} d and not yet purged`);
  if (candidates.length === 0) return 0;

  const { data: clipRows } = await supabase.from("sail_clips").select("clip_key");
  const catalogue = new Set((clipRows ?? []).map((r) => r.clip_key as string));

  const plan: { key: string; objects: { key: string; size: number }[]; row: boolean; audits: number }[] = [];
  for (const c of candidates) {
    const objects = await listPrefix(s3, bucket, `${c.clip_key}/`);
    const { count } = await supabase
      .from("sail_audits").select("*", { count: "exact", head: true }).eq("clip_key", c.clip_key);
    plan.push({ key: c.clip_key, objects, row: catalogue.has(c.clip_key), audits: count ?? 0 });
    const mb = (objects.reduce((a, o) => a + o.size, 0) / 1e6).toFixed(1);
    console.log(
      `  ${c.clip_key}: deleted ${c.deleted_at?.slice(0, 10)} — ${objects.length} object(s), ${mb} MB, ` +
      `catalogue row ${catalogue.has(c.clip_key) ? "yes" : "no"}, ${count ?? 0} audit(s) kept`,
    );
  }
  if (!yes) {
    console.log("dry run: nothing removed. Re-run with --yes to purge the above.");
    return 0;
  }

  for (const p of plan) {
    for (let i = 0; i < p.objects.length; i += 1000) {
      await s3.send(new DeleteObjectsCommand({
        Bucket: bucket,
        Delete: { Objects: p.objects.slice(i, i + 1000).map((o) => ({ Key: o.key })), Quiet: true },
      }));
    }
    if (p.row) {
      const { error: delErr } = await supabase.from("sail_clips").delete().eq("clip_key", p.key);
      if (delErr) throw new Error(`sail_clips delete ${p.key}: ${delErr.message}`);
    }
    const stamp = new Date().toISOString();
    const { error: updErr } = await supabase
      .from("sail_curation")
      .update({ purged_at: stamp, updated_at: stamp, updated_by: "curation:prune" })
      .eq("clip_key", p.key);
    if (updErr) throw new Error(`sail_curation update ${p.key}: ${updErr.message}`);
    const { error: logErr } = await supabase.from("sail_curation_log").insert({
      clip_key: p.key, action: "purge", cuts: null,
      note: `${p.objects.length} objects removed; catalogue row ${p.row ? "removed" : "absent"}; ${p.audits} audits kept`,
      created_by: "curation:prune",
    });
    if (logErr) throw new Error(`sail_curation_log insert ${p.key}: ${logErr.message}`);
    console.log(`purged ${p.key}: ${p.objects.length} object(s)${p.row ? ", catalogue row" : ""}`);
  }
  console.log(`done: ${plan.length} set(s) purged. Re-run \`pnpm curation:export\` and commit configs/curation.yaml.`);
  return 0;
}

main().then(
  (code) => process.exit(code),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
