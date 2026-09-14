import { filterCutObjects } from "@/lib/curation/exportFilter";
import { isDeleted } from "@/lib/curation/logic";
import { CurationRecordSchema, type CurationRecord } from "@/lib/curation/types";
import { isR2Configured, r2Client } from "@/lib/r2";
import {
  GetObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { createClient } from "@supabase/supabase-js";
import { createHash } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

/* Bundle-fetch links: POST {clips: [...]} (or legacy {clip}) returns a
   single URL that downloads the selected clip bundles onto any machine
   (a GPU box) with no dashboard login — `curl -fsSL '<url>' | bash`.
   The URL points at a generated fetch script in the bucket; the script
   holds one presigned GET per bundle object across every selected clip.
   Everything is presigned for 7 days (the R2 maximum), so the link is a
   bearer credential with a hard expiry. Only an authenticated dashboard
   user can mint one (this route sits behind the password gate).

   Curation is honoured: a set deleted in the dashboard is refused (410),
   and the per-frame objects (frames/thermal/seg) inside a set's cut ranges
   are left out of the script. The JSON side files are shipped whole —
   their entries for cut frames are metadata that the Python readers skip
   via configs/curation.yaml. */

const EXPORT_TTL_S = 7 * 24 * 3600; // R2's presign ceiling
const CLIP_KEY = /^[A-Za-z0-9][A-Za-z0-9._-]{2,120}$/;
const MAX_CLIPS = 50;

/** Server-side curation read (anon key suffices: the table is public-read).
    Unconfigured/unreachable store => no curation, never a refused export. */
async function loadCuration(): Promise<Map<string, CurationRecord>> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  const out = new Map<string, CurationRecord>();
  if (!url || !key) return out;
  try {
    const sb = createClient(url, key, { auth: { persistSession: false } });
    const { data, error } = await sb.from("sail_curation").select("*");
    if (error) throw error;
    for (const row of data ?? []) {
      const p = CurationRecordSchema.safeParse(row);
      if (p.success) out.set(p.data.clip_key, p.data);
    }
  } catch (e) {
    console.warn("[export] curation unreachable, exporting uncurated:", e);
  }
  return out;
}

export async function POST(req: NextRequest) {
  if (!isR2Configured()) {
    return NextResponse.json({ error: "R2 not configured" }, { status: 503 });
  }
  let body: { clip?: string; clips?: string[] };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const clips = [...new Set(body.clips ?? (body.clip ? [body.clip] : []))].sort();
  if (
    clips.length === 0 || clips.length > MAX_CLIPS ||
    !clips.every((c) => typeof c === "string" && CLIP_KEY.test(c) && !c.includes(".."))
  ) {
    return NextResponse.json(
      { error: `clips must be 1-${MAX_CLIPS} valid clip keys` }, { status: 400 });
  }

  const curation = await loadCuration();
  const deleted = clips.filter((c) => isDeleted(curation.get(c)));
  if (deleted.length > 0) {
    return NextResponse.json(
      { error: `deleted in the dashboard (restore first): ${deleted.join(", ")}` },
      { status: 410 });
  }

  const s3 = r2Client();
  const bucket = process.env.R2_BUCKET!;
  const listed: { key: string; size: number }[] = [];
  const missing: string[] = [];
  for (const clip of clips) {
    let found = 0;
    let token: string | undefined;
    do {
      const page = await s3.send(new ListObjectsV2Command({
        Bucket: bucket, Prefix: `${clip}/`, ContinuationToken: token,
      }));
      for (const o of page.Contents ?? []) {
        if (o.Key) { listed.push({ key: o.Key, size: o.Size ?? 0 }); found++; }
      }
      token = page.NextContinuationToken;
    } while (token);
    if (found === 0) missing.push(clip);
  }
  if (missing.length > 0) {
    return NextResponse.json(
      { error: `not in the bucket: ${missing.join(", ")}` }, { status: 404 });
  }
  const { kept: keys, cut: cutFiles } = filterCutObjects(listed, curation);

  const totalBytes = keys.reduce((a, k) => a + k.size, 0);
  const mb = (totalBytes / 1e6).toFixed(1);
  // url + destination pairs, fetched 8-wide by xargs. Keys are SEGMENT-safe
  // (no whitespace) and presigned URLs contain none either, so plain
  // word-splitting is sound.
  const pairs: string[] = [];
  for (const { key } of keys) {
    const url = await getSignedUrl(
      s3, new GetObjectCommand({ Bucket: bucket, Key: key }),
      { expiresIn: EXPORT_TTL_S });
    pairs.push(`${url} ${key}`);
  }
  const lines: string[] = [
    "#!/usr/bin/env bash",
    `# ASVProject clip bundle${clips.length > 1 ? "s" : ""}:`,
    ...clips.map((c) => `#   ${c}/`),
    `# ${keys.length} files, ${mb} MB.`,
    ...(cutFiles > 0
      ? [`# ${cutFiles} per-frame file(s) inside dashboard cut ranges omitted (configs/curation.yaml).`]
      : []),
    `# Links expire ${new Date(Date.now() + EXPORT_TTL_S * 1000).toISOString().slice(0, 10)};`,
    "# regenerate from the dashboard's copy-link button after that.",
    "set -euo pipefail",
    `echo "fetching ${clips.length} clip(s): ${keys.length} files, ${mb} MB"`,
    'list=$(mktemp)',
    'trap \'rm -f "$list"\' EXIT',
    "cat > \"$list\" <<'FILE_LIST'",
    ...pairs,
    "FILE_LIST",
    "xargs -n 2 -P 8 sh -c 'curl -fsS --retry 3 --create-dirs -o \"$1\" \"$0\"' < \"$list\"",
    `echo "done: ${clips.map((c) => `./${c}/`).join(" ")}"`,
  ];

  const scriptKey = clips.length === 1
    ? `exports/${clips[0]}.sh`
    : `exports/multi_${createHash("sha256").update(clips.join("\n")).digest("hex").slice(0, 16)}.sh`;
  await s3.send(new PutObjectCommand({
    Bucket: bucket, Key: scriptKey,
    Body: lines.join("\n") + "\n",
    ContentType: "text/x-shellscript",
  }));
  const url = await getSignedUrl(
    s3, new GetObjectCommand({ Bucket: bucket, Key: scriptKey }),
    { expiresIn: EXPORT_TTL_S });
  return NextResponse.json({
    url, files: keys.length, bytes: totalBytes, clips: clips.length,
    cut_files: cutFiles,
    command: `curl -fsSL '${url}' | bash`,
  });
}
