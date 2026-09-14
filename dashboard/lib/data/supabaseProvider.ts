import { createClient } from "@/lib/supabase/client";
import {
  ClipMetaSchema,
  ClipSummarySchema,
  type Box,
  type ClipMeta,
  type ClipSummary,
  type Instance,
  type LabelRecord,
  type RadarFrames,
  type SectorRecord,
  safeTs,
} from "@/lib/types";
import type { DataProvider, FrameKind } from "./provider";

export const BUCKET = "asvproject-clips";

/** Supabase-backed provider.
 *  - catalogue rows in public.sail_clips (meta jsonb per clip)
 *  - bundle assets from NEXT_PUBLIC_ASSET_BASE when set (the /api/assets
 *    route presigning the PRIVATE R2 bucket), else the Supabase Storage
 *    bucket. Bundle layout is identical everywhere (tools/bake_demo.py).
 */
export function createSupabaseProvider(): DataProvider {
  const supabase = createClient();
  const assetBase = process.env.NEXT_PUBLIC_ASSET_BASE;

  function publicUrl(path: string): string {
    if (assetBase) return `${assetBase.replace(/\/$/, "")}/${path}`;
    return supabase.storage.from(BUCKET).getPublicUrl(path).data.publicUrl;
  }

  // Signed-URL resolution via the route's json mode, cached below the
  // signing quantisation window so entries never outlive their signature.
  const signedCache = new Map<string, { url: Promise<string>; born: number }>();
  const SIGNED_TTL_MS = 45 * 60 * 1000;

  function resolveUrl(path: string): Promise<string> {
    if (!assetBase) return Promise.resolve(publicUrl(path));
    const hit = signedCache.get(path);
    if (hit && Date.now() - hit.born < SIGNED_TTL_MS) return hit.url;
    const url = fetch(`${publicUrl(path)}?json=1`).then(async (res) => {
      if (!res.ok) throw new Error(`asset sign ${path}: HTTP ${res.status}`);
      return ((await res.json()) as { url: string }).url;
    });
    url.catch(() => signedCache.delete(path)); // don't cache failures
    signedCache.set(path, { url, born: Date.now() });
    if (signedCache.size > 800) {
      const oldest = signedCache.keys().next().value;
      if (oldest) signedCache.delete(oldest);
    }
    return url;
  }

  // Labels are mutable (teacher republishes) and audit-critical, so they
  // revalidate (no-cache -> conditional GET, a 304 when unchanged); the
  // other side files are immutable per bake and keep force-cache.
  async function storageJson<T>(path: string, cache: RequestCache = "force-cache"): Promise<T> {
    const res = await fetch(await resolveUrl(path), { cache });
    if (!res.ok) throw new Error(`storage asset ${path}: HTTP ${res.status}`);
    return (await res.json()) as T;
  }

  return {
    mode: "supabase",
    async listClips(): Promise<ClipSummary[]> {
      const { data, error } = await supabase
        .from("sail_clips")
        .select("clip_key, summary")
        .order("clip_key");
      if (error) throw error;
      return (data ?? []).map((row) => ClipSummarySchema.parse(row.summary));
    },
    async getMeta(clipKey: string): Promise<ClipMeta> {
      const { data, error } = await supabase
        .from("sail_clips")
        .select("meta")
        .eq("clip_key", clipKey)
        .single();
      if (error) throw error;
      return ClipMetaSchema.parse(data.meta);
    },
    getSectors(clipKey: string) {
      return storageJson<Record<string, SectorRecord>>(
        `${clipKey}/sectors.json`,
      );
    },
    getRadar(clipKey: string) {
      return storageJson<RadarFrames>(`${clipKey}/radar.json`);
    },
    getBoxes(clipKey: string) {
      return storageJson<Record<string, Box[]>>(`${clipKey}/boxes.json`);
    },
    getInstances(clipKey: string) {
      return storageJson<Record<string, Instance[]>>(
        `${clipKey}/instances.json`,
      );
    },
    getLabels(clipKey: string) {
      return storageJson<Record<string, LabelRecord>>(
        `${clipKey}/labels.json`,
        "no-cache",
      );
    },
    frameUrl(clipKey: string, kind: FrameKind, ts: string): string {
      const ext = kind === "seg" || kind === "thermal_hd" ? "png" : "jpg";
      return publicUrl(`${clipKey}/${kind}/ts=${safeTs(ts)}.${ext}`);
    },
    resolveAssetUrl(clipKey: string, kind: FrameKind, ts: string) {
      const ext = kind === "seg" || kind === "thermal_hd" ? "png" : "jpg";
      return resolveUrl(`${clipKey}/${kind}/ts=${safeTs(ts)}.${ext}`);
    },
  };
}
