/* Shared bootstrap for the curation CLIs: load dashboard/.env.local into
 * process.env (never overriding what the shell already set), and build the
 * Supabase + R2 clients the tools need. Kept dependency-free beyond what the
 * ingest tools already use. */

import { S3Client } from "@aws-sdk/client-s3";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DASHBOARD_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export const REPO_ROOT = path.resolve(DASHBOARD_DIR, "..");

export function loadEnvLocal(): void {
  const p = path.join(DASHBOARD_DIR, ".env.local");
  if (!existsSync(p)) return;
  for (const line of readFileSync(p, "utf8").split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const i = t.indexOf("=");
    if (i < 0) continue;
    const k = t.slice(0, i).trim();
    const v = t.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    if (!(k in process.env)) process.env[k] = v;
  }
}

export function supabaseAdmin(opts: { requireServiceRole?: boolean } = {}): SupabaseClient {
  const url = process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL;
  const service = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  const key = service ?? (opts.requireServiceRole ? undefined : anon);
  if (!url || !key) {
    throw new Error(
      opts.requireServiceRole
        ? "NEXT_PUBLIC_SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required (dashboard/.env.local)"
        : "NEXT_PUBLIC_SUPABASE_URL + a Supabase key required (dashboard/.env.local)",
    );
  }
  return createClient(url, key, { auth: { persistSession: false } });
}

export function r2(): { s3: S3Client; bucket: string } {
  const { R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET } = process.env;
  if (!R2_ACCOUNT_ID || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY || !R2_BUCKET) {
    throw new Error("R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET required");
  }
  return {
    s3: new S3Client({
      region: "auto",
      endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
      credentials: { accessKeyId: R2_ACCESS_KEY_ID, secretAccessKey: R2_SECRET_ACCESS_KEY },
    }),
    bucket: R2_BUCKET,
  };
}

export function flag(name: string): boolean {
  return process.argv.includes(name);
}

export function option(name: string): string | undefined {
  const i = process.argv.indexOf(name);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  const eq = process.argv.find((a) => a.startsWith(`${name}=`));
  return eq ? eq.slice(name.length + 1) : undefined;
}
