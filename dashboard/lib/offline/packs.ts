import { keptFrames } from "@/lib/curation/logic";
import type { CurationRecord } from "@/lib/curation/types";
import type { DataProvider, FrameKind } from "@/lib/data/provider";
import type { AuditRecord, ClipMeta, LabelRecord } from "@/lib/types";
import { idbDelete, idbGet, idbGetAll, idbPut, STORE_PACKS } from "./db";

/* Offline packs: everything the annotate stage needs for a set of frames,
   stored in the browser — image bytes in Cache Storage (one cache per pack,
   keyed by the SAME same-origin asset URL the page requests, so the service
   worker answers `<img src>` loads from the pack), JSON side files + an
   audit snapshot in IndexedDB. Packs never include deleted sets or cut
   frames (curation is applied at pack time).

   Packs live in the browser profile: they are not isolated between users
   sharing one profile, and the browser may evict them unless storage is
   persisted (we ask). */

export const PACK_CACHE_PREFIX = "asvproject-pack:";
export const PACK_EVENT = "asvproject-packs";

/** Rough per-frame budget for the size estimate (432×324 JPEG ≈ 50–100 KB
    fisheye, 160×120 thermal ≈ 6 KB). */
export const EST_FISHEYE_BYTES = 75_000;
export const EST_THERMAL_BYTES = 6_000;

export interface PackFrame {
  clipKey: string;
  ts: string;
  thermal: boolean;
}

export interface PackedClip {
  title: string;
  meta: ClipMeta;
  labels: Record<string, LabelRecord>;
  audits: AuditRecord[];
}

export interface PackManifest {
  id: string;
  kind: "plan" | "set";
  name: string;
  createdAt: string;
  status: "downloading" | "ready" | "partial" | "failed";
  frames: PackFrame[];
  clips: Record<string, PackedClip>;
  bytes: number;
  done: number;
  failed: number;
  error?: string;
}

export function packCacheName(id: string): string {
  return `${PACK_CACHE_PREFIX}${id}`;
}

export function estimateBytes(frames: readonly { thermal: boolean }[]): number {
  return frames.reduce((a, f) => a + EST_FISHEYE_BYTES + (f.thermal ? EST_THERMAL_BYTES : 0), 0);
}

function notify(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(PACK_EVENT));
}

export function subscribePacks(cb: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(PACK_EVENT, cb);
  return () => window.removeEventListener(PACK_EVENT, cb);
}

export async function listPacks(): Promise<PackManifest[]> {
  return (await idbGetAll<PackManifest>(STORE_PACKS)).sort((a, b) =>
    b.createdAt.localeCompare(a.createdAt),
  );
}

export async function getPack(id: string): Promise<PackManifest | null> {
  return idbGet<PackManifest>(STORE_PACKS, id);
}

export async function deletePack(id: string): Promise<void> {
  if (typeof caches !== "undefined") await caches.delete(packCacheName(id));
  await idbDelete(STORE_PACKS, id);
  notify();
}

/** The packed clip data for a clip key, from the newest pack holding it. */
export async function packedClip(clipKey: string): Promise<PackedClip | null> {
  for (const p of await listPacks()) {
    const c = p.clips[clipKey];
    if (c) return c;
  }
  return null;
}

/** Refresh a clip's JSON side files (meta/labels/audits) in every pack
    that holds it. Called after each successful ONLINE load of the clip, so
    a later network-failure fallback serves current teacher labels instead
    of whatever happened to be live when the pack was downloaded. */
export async function refreshPackedClip(clipKey: string, data: PackedClip): Promise<void> {
  for (const p of await listPacks()) {
    if (!p.clips[clipKey]) continue;
    p.clips[clipKey] = data;
    await idbPut(STORE_PACKS, p);
  }
  notify();
}

/** Which frames of a clip any pack holds (for the "packed" markers). */
export async function packedFrames(clipKey: string): Promise<Set<string>> {
  const out = new Set<string>();
  for (const p of await listPacks()) {
    for (const f of p.frames) if (f.clipKey === clipKey) out.add(f.ts);
  }
  return out;
}

/** Frames of a set that a pack may include: not deleted, not cut. */
export function packableFrames(
  clipKey: string,
  meta: ClipMeta,
  record: CurationRecord | undefined | null,
): PackFrame[] {
  return keptFrames(meta.frames.filter((f) => f.fisheye), record).map((f) => ({
    clipKey,
    ts: f.ts,
    thermal: f.thermal,
  }));
}

/** Plan frames filtered by curation (deleted sets / cut frames dropped)
    and annotated with whether a thermal frame exists. */
export function packablePlanFrames(
  plan: readonly { clipKey: string; ts: string }[],
  metas: ReadonlyMap<string, ClipMeta>,
  records: ReadonlyMap<string, CurationRecord>,
): { frames: PackFrame[]; dropped: number } {
  const frames: PackFrame[] = [];
  let dropped = 0;
  for (const f of plan) {
    const meta = metas.get(f.clipKey);
    if (!meta) {
      dropped++;
      continue;
    }
    const kept = new Set(packableFrames(f.clipKey, meta, records.get(f.clipKey)).map((k) => k.ts));
    if (!kept.has(f.ts)) {
      dropped++;
      continue;
    }
    const entry = meta.frames.find((e) => e.ts === f.ts);
    frames.push({ clipKey: f.clipKey, ts: f.ts, thermal: Boolean(entry?.thermal) });
  }
  return { frames, dropped };
}

export interface DownloadProgress {
  done: number;
  failed: number;
  total: number;
  bytes: number;
}

/** Fetch one asset via the provider's CORS-safe resolver and store its
    bytes under the page-facing URL. Returns the byte count. */
async function fetchInto(
  cache: Cache,
  provider: DataProvider,
  clipKey: string,
  kind: FrameKind,
  ts: string,
): Promise<number> {
  const pageUrl = new URL(provider.frameUrl(clipKey, kind, ts), location.origin).toString();
  const src = await provider.resolveAssetUrl(clipKey, kind, ts);
  // cache: "reload" — the viewer may already hold this exact signed URL in
  // the HTTP cache as a no-CORS <img> response (no Access-Control-Allow-
  // Origin header); a CORS fetch served from that entry fails. Go to the
  // network so the response carries its CORS headers.
  const res = await fetch(src, { mode: "cors", cache: "reload" });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${kind} ${ts}`);
  const len = Number(res.headers.get("content-length") ?? 0);
  const body = await res.arrayBuffer();
  await cache.put(
    new Request(pageUrl),
    new Response(body, {
      status: 200,
      headers: {
        "content-type": res.headers.get("content-type") ?? (kind === "seg" ? "image/png" : "image/jpeg"),
        "cache-control": "private, max-age=31536000, immutable",
      },
    }),
  );
  return len || body.byteLength;
}

/** Build a pack: JSON side files first (so the manifest is useful even if
    images fail), then images with bounded concurrency, progress reported
    per frame. Persistent storage is requested once per profile. */
export async function downloadPack(
  provider: DataProvider,
  spec: { id: string; kind: PackManifest["kind"]; name: string; frames: PackFrame[] },
  listAudits: (clipKey: string) => Promise<AuditRecord[]>,
  onProgress?: (p: DownloadProgress) => void,
  concurrency = 6,
): Promise<PackManifest> {
  try {
    await navigator.storage?.persist?.();
  } catch {
    /* not supported: packs still work, just evictable */
  }
  const manifest: PackManifest = {
    id: spec.id,
    kind: spec.kind,
    name: spec.name,
    createdAt: new Date().toISOString(),
    status: "downloading",
    frames: spec.frames,
    clips: {},
    bytes: 0,
    done: 0,
    failed: 0,
  };
  const clipKeys = [...new Set(spec.frames.map((f) => f.clipKey))];
  for (const key of clipKeys) {
    const [meta, labels, audits] = await Promise.all([
      provider.getMeta(key),
      provider.getLabels(key).catch(() => ({}) as Record<string, LabelRecord>),
      listAudits(key).catch(() => [] as AuditRecord[]),
    ]);
    manifest.clips[key] = { title: meta.title, meta, labels, audits };
    manifest.bytes += JSON.stringify(meta).length + JSON.stringify(labels).length;
  }
  await idbPut(STORE_PACKS, manifest);
  notify();

  const cache = await caches.open(packCacheName(spec.id));
  const queue = [...spec.frames];
  const total = spec.frames.length;
  let lastSave = 0;
  const workers = Array.from({ length: Math.max(1, concurrency) }, async () => {
    for (;;) {
      const f = queue.shift();
      if (!f) return;
      try {
        manifest.bytes += await fetchInto(cache, provider, f.clipKey, "frames", f.ts);
        if (f.thermal) {
          try {
            manifest.bytes += await fetchInto(cache, provider, f.clipKey, "thermal", f.ts);
          } catch {
            /* thermal is optional: the stage shows the fisheye either way */
          }
        }
        manifest.done++;
      } catch (e) {
        manifest.failed++;
        manifest.error = String((e as Error)?.message ?? e);
      }
      onProgress?.({ done: manifest.done, failed: manifest.failed, total, bytes: manifest.bytes });
      const now = Date.now();
      if (now - lastSave > 1500) {
        lastSave = now;
        await idbPut(STORE_PACKS, { ...manifest });
        notify();
      }
    }
  });
  await Promise.all(workers);
  manifest.status = manifest.failed === 0 ? "ready" : manifest.done > 0 ? "partial" : "failed";
  await idbPut(STORE_PACKS, manifest);
  notify();
  return manifest;
}

/** Storage estimate (bytes) for the settings readout; null if unsupported. */
export async function storageEstimate(): Promise<{ usage: number; quota: number; persisted: boolean } | null> {
  try {
    const est = await navigator.storage?.estimate?.();
    const persisted = (await navigator.storage?.persisted?.()) ?? false;
    if (!est) return null;
    return { usage: est.usage ?? 0, quota: est.quota ?? 0, persisted };
  } catch {
    return null;
  }
}
