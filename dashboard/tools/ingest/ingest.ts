/* Upload a baked clip bundle to Supabase.
 *
 * Reads the bundle directory produced by dashboard/tools/bake_demo.py
 * (default: dashboard/public/demo), upserts one row per clip into
 * public.sail_clips and uploads every asset into the public Storage bucket
 * `asvproject-clips` with the same relative paths the demo provider uses —
 * so the web app needs zero code changes between modes.
 *
 * Usage (from dashboard/):
 *   SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... pnpm ingest [bundleDir]
 */

import { createClient } from "@supabase/supabase-js";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const BUCKET = "asvproject-clips";

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

async function main(): Promise<number> {
  const url =
    process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) {
    console.error(
      "SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) and SUPABASE_SERVICE_ROLE_KEY are required",
    );
    return 1;
  }
  const bundleDir = path.resolve(
    process.argv[2] ?? path.join(import.meta.dirname, "..", "..", "public", "demo"),
  );
  const supabase = createClient(url, serviceKey, {
    auth: { persistSession: false },
  });

  // Public bucket: public read is built in, no storage.objects policy needed.
  const { error: bucketError } = await supabase.storage.createBucket(BUCKET, {
    public: true,
  });
  if (bucketError && !/already exists/i.test(bucketError.message)) {
    throw new Error(`createBucket failed: ${bucketError.message}`);
  }

  const catalogue = JSON.parse(
    await readFile(path.join(bundleDir, "clips.json"), "utf8"),
  ) as { clips: { clip_id: string }[] };

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
    if (rowError) throw new Error(`sail_clips upsert failed: ${rowError.message}`);

    const files = await walk(clipDir);
    let uploaded = 0;
    for (const file of files) {
      const rel = path.relative(bundleDir, file).split(path.sep).join("/");
      const body = await readFile(file);
      const { error } = await supabase.storage
        .from(BUCKET)
        .upload(rel, body, {
          upsert: true,
          contentType:
            CONTENT_TYPES[path.extname(file)] ?? "application/octet-stream",
          cacheControl: "31536000", // bundle assets are immutable per bake
        });
      if (error) throw new Error(`upload ${rel} failed: ${error.message}`);
      uploaded++;
      if (uploaded % 50 === 0) {
        console.log(`${clipKey}: ${uploaded}/${files.length} files`);
      }
    }
    console.log(`${clipKey}: ${uploaded} files uploaded, catalogue row upserted`);
  }
  console.log("done");
  return 0;
}

main().then(
  (code) => process.exit(code),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
