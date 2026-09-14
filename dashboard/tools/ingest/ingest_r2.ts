/* Upload a baked bundle to the PRIVATE R2 bucket + upsert catalogue rows in
 * Supabase (public.sail_clips). The web app then serves assets through the
 * signed /api/assets route; nothing in R2 is publicly reachable.
 *
 * Skips unchanged files by comparing sizes with a bucket listing, so re-runs
 * after an incremental bake only upload what changed.
 *
 * Usage (from dashboard/, env from .env.local):
 *   pnpm ingest:r2 [bundleDir]        # default: /Volumes/ROS2_SSD/asvproject/dashboard_bundle
 */

import {
  DeleteObjectsCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { createClient } from "@supabase/supabase-js";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { gzipSync } from "node:zlib";

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".jpg": "image/jpeg",
  ".png": "image/png",
};

async function walk(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...(await walk(full)));
    else files.push(full);
  }
  return files;
}

async function listBucketSizes(s3: S3Client, bucket: string) {
  const sizes = new Map<string, number>();
  let token: string | undefined;
  do {
    const page = await s3.send(
      new ListObjectsV2Command({ Bucket: bucket, ContinuationToken: token }),
    );
    for (const obj of page.Contents ?? []) {
      if (obj.Key) sizes.set(obj.Key, obj.Size ?? -1);
    }
    token = page.NextContinuationToken;
  } while (token);
  return sizes;
}

async function main(): Promise<number> {
  const { R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET } =
    process.env;
  const supabaseUrl =
    process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!R2_ACCOUNT_ID || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY || !R2_BUCKET) {
    console.error("R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET required");
    return 1;
  }
  if (!supabaseUrl || !serviceKey) {
    console.error("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required (catalogue rows)");
    return 1;
  }
  const positional = process.argv
    .slice(2)
    .filter((a) => a !== "--" && !a.startsWith("--"));
  const bundleDir = path.resolve(
    positional[0] ?? "/Volumes/ROS2_SSD/asvproject/dashboard_bundle",
  );
  const s3 = new S3Client({
    region: "auto",
    endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: R2_ACCESS_KEY_ID,
      secretAccessKey: R2_SECRET_ACCESS_KEY,
    },
  });
  const supabase = createClient(supabaseUrl, serviceKey, {
    auth: { persistSession: false },
  });

  console.log("listing bucket…");
  const existing = await listBucketSizes(s3, R2_BUCKET);
  console.log(`bucket holds ${existing.size} objects`);

  const catalogue = JSON.parse(
    await readFile(path.join(bundleDir, "clips.json"), "utf8"),
  ) as { clips: { clip_id: string }[] };

  let uploaded = 0;
  let skipped = 0;
  for (const summary of catalogue.clips) {
    const clipKey = summary.clip_id.replace("/", "__");
    const clipDir = path.join(bundleDir, clipKey);
    const meta = JSON.parse(
      await readFile(path.join(clipDir, "meta.json"), "utf8"),
    );

    const { error: rowError } = await supabase.from("sail_clips").upsert({
      clip_key: clipKey,
      clip_id: summary.clip_id,
      summary,
      meta,
      updated_at: new Date().toISOString(),
    });
    if (rowError) throw new Error(`sail_clips upsert: ${rowError.message}`);

    const files = await walk(clipDir);
    const queue = [...files];
    const CONCURRENCY = 48;
    await Promise.all(
      Array.from({ length: CONCURRENCY }, async () => {
        for (;;) {
          const file = queue.shift();
          if (!file) return;
          const rel = path.relative(bundleDir, file).split(path.sep).join("/");
          const size = (await stat(file)).size;
          if (existing.get(rel) === size) {
            skipped++;
            continue;
          }
          let body = await readFile(file);
          // labels.json can reach tens of MB (union teachers + mask polygons);
          // store it gzip'd with Content-Encoding so browsers decode
          // transparently and the annotate page loads in ~1 s instead of
          // failing over to a stale offline pack. NB the size-skip above
          // compares against the *stored* (gzip'd) size, so re-ingesting an
          // unchanged bundle re-uploads labels.json once — acceptable.
          const gzip = path.basename(file) === "labels.json";
          if (gzip) body = gzipSync(body, { level: 8 });
          await s3.send(
            new PutObjectCommand({
              Bucket: R2_BUCKET,
              Key: rel,
              Body: body,
              ContentType:
                CONTENT_TYPES[path.extname(file)] ?? "application/octet-stream",
              ...(gzip ? { ContentEncoding: "gzip" } : {}),
            }),
          );
          uploaded++;
          if (uploaded % 1000 === 0) console.log(`uploaded ${uploaded}…`);
        }
      }),
    );
    console.log(`${clipKey}: catalogue row upserted (${files.length} files)`);
  }
  console.log(`done: ${uploaded} uploaded, ${skipped} unchanged`);

  // --prune: remove R2 objects + catalogue rows for clip keys no longer in
  // the bundle (e.g. per-chunk clips replaced by concatenated activities).
  if (process.argv.includes("--prune")) {
    const valid = new Set(
      catalogue.clips.map((c) => c.clip_id.replace("/", "__")),
    );
    const stale = [...existing.keys()].filter(
      (k) => !valid.has(k.split("/")[0]!),
    );
    console.log(`pruning ${stale.length} stale objects…`);
    for (let i = 0; i < stale.length; i += 1000) {
      await s3.send(
        new DeleteObjectsCommand({
          Bucket: R2_BUCKET,
          Delete: {
            Objects: stale.slice(i, i + 1000).map((Key) => ({ Key })),
            Quiet: true,
          },
        }),
      );
    }
    const { data: rows, error } = await supabase
      .from("sail_clips")
      .select("clip_key");
    if (error) throw new Error(`sail_clips select: ${error.message}`);
    const staleRows = (rows ?? [])
      .map((r) => r.clip_key as string)
      .filter((k) => !valid.has(k));
    if (staleRows.length) {
      const { error: delError } = await supabase
        .from("sail_clips")
        .delete()
        .in("clip_key", staleRows);
      if (delError) throw new Error(`sail_clips prune: ${delError.message}`);
    }
    console.log(
      `pruned ${stale.length} objects, ${staleRows.length} catalogue rows`,
    );
  }
  return 0;
}

main().then(
  (code) => process.exit(code),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
