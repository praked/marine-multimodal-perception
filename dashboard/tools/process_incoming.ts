/* Workstation processor for browser-uploaded capture sessions.

   Watches `incoming/<session>/` in the private bucket (queued by the
   dashboard's /upload page, tracked in sail_ingests) and runs the SAME
   pipeline as tools/ingest_session.py from the download step onward:

     download -> validate (ingest_session.validate) -> bake --force --only
     -> enrich -> ingest_r2 -> status=ready (or failed, with the report)

   Usage (from dashboard/):
     pnpm process:incoming            # process queued sessions once
     pnpm process:incoming --watch    # poll every 60 s

   Needs: the ROS2_SSD mounted (sessions land in the canonical captures
   tree) and the repo venv for the python steps. Env comes from
   .env.local automatically. */

import {
  GetObjectCommand,
  ListObjectsV2Command,
  S3Client,
} from "@aws-sdk/client-s3";
import { createClient } from "@supabase/supabase-js";
import { execFileSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

// .env.local loader (same minimal parse ingest_r2 relies on via bash)
try {
  const env = readFileSync(path.resolve(__dirname, "..", ".env.local"), "utf8");
  for (const line of env.split("\n")) {
    const m = line.match(/^([A-Z0-9_]+)=("?)(.*)\2$/);
    if (m && !(m[1]! in process.env)) process.env[m[1]!] = m[3]!;
  }
} catch {
  // fine — env may already be exported
}

const {
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET,
  NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
  NEXT_PUBLIC_SUPABASE_ANON_KEY,
} = process.env;
if (!R2_ACCOUNT_ID || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY || !R2_BUCKET) {
  console.error("R2_* env required (source .env.local)");
  process.exit(1);
}
const SSD = "/Volumes/ROS2_SSD/asvproject";
const REPO = path.resolve(__dirname, "..", "..");
const PY = path.join(REPO, ".venv", "bin", "python");

const s3 = new S3Client({
  region: "auto",
  endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: { accessKeyId: R2_ACCESS_KEY_ID, secretAccessKey: R2_SECRET_ACCESS_KEY },
});
const supabase = NEXT_PUBLIC_SUPABASE_URL
  ? createClient(NEXT_PUBLIC_SUPABASE_URL,
      SUPABASE_SERVICE_ROLE_KEY ?? NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "")
  : null;

async function setStatus(session: string, status: string, report?: object) {
  if (!supabase) return;
  await supabase.from("sail_ingests").upsert(
    { session, status, ...(report ? { report } : {}),
      updated_at: new Date().toISOString() },
    { onConflict: "session" });
}

async function queuedSessions(): Promise<string[]> {
  if (supabase) {
    const { data } = await supabase
      .from("sail_ingests").select("session").eq("status", "uploaded");
    if (data) return data.map((r) => r.session as string);
  }
  // fallback: whatever sits under incoming/ that isn't on disk yet
  const out = new Set<string>();
  const list = await s3.send(new ListObjectsV2Command({
    Bucket: R2_BUCKET, Prefix: "incoming/", Delimiter: "/" }));
  for (const p of list.CommonPrefixes ?? []) {
    const session = p.Prefix!.slice("incoming/".length).replace(/\/$/, "");
    if (session) out.add(session);
  }
  return [...out];
}

async function downloadSession(session: string): Promise<string> {
  const dest = path.join(SSD, "captures", session);
  mkdirSync(dest, { recursive: true });
  let token: string | undefined;
  do {
    const page = await s3.send(new ListObjectsV2Command({
      Bucket: R2_BUCKET, Prefix: `incoming/${session}/`,
      ContinuationToken: token }));
    for (const obj of page.Contents ?? []) {
      const name = path.basename(obj.Key!);
      const file = path.join(dest, name);
      if (existsSync(file)) continue;
      const r = await s3.send(new GetObjectCommand({
        Bucket: R2_BUCKET, Key: obj.Key! }));
      await pipeline(r.Body as Readable, createWriteStream(file));
      console.log(`  ↓ ${name} (${((obj.Size ?? 0) / 1e6).toFixed(1)} MB)`);
    }
    token = page.NextContinuationToken;
  } while (token);
  return dest;
}

function run(cmd: string, args: string[], cwd = REPO): string {
  console.log(`  $ ${cmd} ${args.join(" ")}`);
  return execFileSync(cmd, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] });
}

async function processSession(session: string): Promise<void> {
  console.log(`=== ${session}`);
  await setStatus(session, "processing");
  try {
    const dir = await downloadSession(session);
    // 1. the same validation ingest_session runs after its rsync
    run(PY, ["-c", [
      "import sys; sys.path.insert(0, 'dashboard/tools')",
      "from pathlib import Path",
      "from ingest_session import validate",
      `validate(Path(${JSON.stringify(dir)}))`,
    ].join("\n")]);
    // 2. same chain as ingest_session.py: bake this session, re-enrich
    //    (rebakes drop the enrichment block), publish with prune
    run(PY, ["dashboard/tools/bake_corpus.py", "--only", session]);
    run(PY, ["dashboard/tools/enrich_bundle.py"]);
    run("npx", ["tsx", "tools/ingest/ingest_r2.ts", "--prune"],
      path.join(REPO, "dashboard"));
    await setStatus(session, "ready", {
      summary: "validated, baked, enriched, published",
      finished: new Date().toISOString(),
    });
    console.log(`=== ${session}: READY`);
  } catch (e) {
    await setStatus(session, "failed", { summary: String(e).slice(0, 2000) });
    console.error(`=== ${session}: FAILED — ${e}`);
  }
}

async function main() {
  if (!existsSync(path.join(SSD, "captures"))) {
    console.error("ROS2_SSD not mounted — refusing to process");
    process.exit(1);
  }
  const watch = process.argv.includes("--watch");
  for (;;) {
    const sessions = await queuedSessions();
    if (sessions.length === 0) console.log("nothing queued");
    for (const s of sessions) await processSession(s);
    if (!watch) break;
    await new Promise((r) => setTimeout(r, 60000));
  }
}

void main();
