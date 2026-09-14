/* Upload the swarm-crop review assets to the private R2 bucket.
 *
 *   node --env-file=.env.local tools/swarm_crops/upload_crops.mjs \
 *     --dir <scratch>/swarm_crops/2026-08-26 [--set 2026-08-26] [--force]
 *
 * Uploads crops/<sha1>.jpg under swarm_crops/<set>/crops/, plus the full
 * manifest.jsonl and a compact manifest.json ([crop_id, chunk, ts, w, h]
 * rows, order preserved) the /crops page fetches. Resume-safe: already
 * uploaded keys are listed first and skipped (crop content is deterministic
 * per id); --force re-uploads everything.
 */

import {
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { createReadStream } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const DIR = opt("dir");
const SET = opt("set", "2026-08-26");
const FORCE = args.includes("--force");
const MANIFEST_ONLY = args.includes("--manifest-only");
if (!DIR) {
  console.error("--dir <extract output dir> is required");
  process.exit(1);
}
for (const k of ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]) {
  if (!process.env[k]) {
    console.error(`${k} missing — run with node --env-file=.env.local`);
    process.exit(1);
  }
}

const client = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});
const BUCKET = process.env.R2_BUCKET;
const PREFIX = `swarm_crops/${SET}/`;

async function listExisting() {
  const keys = new Set();
  let token;
  do {
    const res = await client.send(
      new ListObjectsV2Command({
        Bucket: BUCKET,
        Prefix: PREFIX,
        ContinuationToken: token,
      }),
    );
    for (const o of res.Contents ?? []) keys.add(o.Key);
    token = res.IsTruncated ? res.NextContinuationToken : undefined;
  } while (token);
  return keys;
}

async function put(key, body, contentType) {
  await client.send(
    new PutObjectCommand({ Bucket: BUCKET, Key: key, Body: body, ContentType: contentType }),
  );
}

// ---- manifest --------------------------------------------------------------
const manifestPath = path.join(DIR, "manifest.jsonl");
const jsonl = await readFile(manifestPath, "utf8");
const rows = jsonl
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));
// compact row: [crop_id, chunk, ts, crop_w, crop_h, window, xyxy]
// window = the padded crop rect in frame pixels (the jpg IS this region);
// xyxy = the source detection box, normalised on the 864x648 frame.
const compact = rows.map((r) => [
  r.crop_id,
  r.chunk,
  r.ts,
  r.crop_w,
  r.crop_h,
  r.window ?? null,
  r.xyxy ? r.xyxy.map((v) => Math.round(v * 1e4) / 1e4) : null,
]);
console.log(`${rows.length} manifest rows`);

// ---- crops -----------------------------------------------------------------
const cropsDir = path.join(DIR, "crops");
if (MANIFEST_ONLY) {
  await put(`${PREFIX}manifest.json`, JSON.stringify(compact), "application/json");
  await put(`${PREFIX}manifest.jsonl`, jsonl, "application/jsonl");
  console.log("manifests re-uploaded (crops untouched)");
  process.exit(0);
}
const files = (await readdir(cropsDir)).filter((f) => f.endsWith(".jpg"));
const existing = FORCE ? new Set() : await listExisting();
const todo = files.filter((f) => !existing.has(`${PREFIX}crops/${f}`));
console.log(`${files.length} crop files, ${todo.length} to upload (${existing.size} keys already present)`);

let done = 0;
let failed = 0;
const CONCURRENCY = 40;
async function worker(queue) {
  for (;;) {
    const f = queue.pop();
    if (!f) return;
    const key = `${PREFIX}crops/${f}`;
    let ok = false;
    for (let attempt = 0; attempt < 3 && !ok; attempt++) {
      try {
        await put(key, createReadStream(path.join(cropsDir, f)), "image/jpeg");
        ok = true;
      } catch (e) {
        if (attempt === 2) {
          failed++;
          console.error(`FAILED ${f}: ${e.message ?? e}`);
        } else {
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
        }
      }
    }
    if (ok && ++done % 2000 === 0) console.log(`  ${done}/${todo.length}`);
  }
}
const queue = [...todo];
await Promise.all(Array.from({ length: CONCURRENCY }, () => worker(queue)));
console.log(`crops uploaded: ${done}, failed: ${failed}`);
if (failed > 0) process.exit(1);

// manifests last, so a page that sees the manifest can see every crop
await put(`${PREFIX}manifest.json`, JSON.stringify(compact), "application/json");
await put(`${PREFIX}manifest.jsonl`, jsonl, "application/jsonl");
const { size } = await stat(manifestPath);
console.log(
  `manifests uploaded: compact ${(JSON.stringify(compact).length / 1e6).toFixed(1)} MB, full ${(size / 1e6).toFixed(1)} MB`,
);
console.log("done");
