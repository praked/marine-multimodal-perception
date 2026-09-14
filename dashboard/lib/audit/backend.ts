import { isSupabaseConfigured } from "@/lib/data/provider";
import { packedClip } from "@/lib/offline/packs";
import {
  classifyError,
  enqueueAudit,
  isNetworkError,
  queuedAudits,
  syncQueue,
  type SyncReport,
  queuedCount,
} from "@/lib/offline/queue";
import { createClient } from "@/lib/supabase/client";
import { AuditRecordSchema, type AuditRecord } from "@/lib/types";

/* Append-only audit log, mirroring the repo's JSONL discipline: records are
   never updated or deleted; readers fold to latest-per-frame. Demo mode
   persists to localStorage so the whole annotation loop works with no
   credentials; Supabase mode appends to public.sail_audits.

   Offline: when the Supabase insert cannot reach the network the record
   goes to the IndexedDB queue (lib/offline/queue) and `syncPendingAudits`
   replays it later — the client uuid is the row's primary key, so a retry
   can never duplicate. Listing falls back to the offline pack's audit
   snapshot and always merges the still-queued records, so the stage shows
   what the annotator did regardless of connectivity. */

export interface AuditBackend {
  readonly mode: "local" | "supabase";
  list(clipKey: string): Promise<AuditRecord[]>;
  append(record: AuditRecord): Promise<void>;
}

const LS_KEY = "asvproject.audits.v1";

function readLocal(): AuditRecord[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as unknown[];
    return arr
      .map((r) => AuditRecordSchema.safeParse(r))
      .filter((p) => p.success)
      .map((p) => p.data);
  } catch {
    return [];
  }
}

export function createLocalBackend(): AuditBackend {
  return {
    mode: "local",
    async list(clipKey: string) {
      return readLocal().filter((r) => r.clip_key === clipKey);
    },
    async append(record: AuditRecord) {
      const all = readLocal();
      all.push(record);
      window.localStorage.setItem(LS_KEY, JSON.stringify(all));
    },
  };
}

/** Merge records by id (queued copies win nothing: same content). */
export function mergeAudits(a: AuditRecord[], b: AuditRecord[]): AuditRecord[] {
  const seen = new Map<string, AuditRecord>();
  for (const r of [...a, ...b]) if (!seen.has(r.id)) seen.set(r.id, r);
  return [...seen.values()].sort((x, y) => x.created_at.localeCompare(y.created_at));
}

/** The raw online insert (throws the Supabase error object on failure). */
async function insertSupabase(
  supabase: ReturnType<typeof createClient>,
  record: AuditRecord,
): Promise<void> {
  const { error } = await supabase.from("sail_audits").insert({
    id: record.id,
    clip_key: record.clip_key,
    frame_ts: record.frame_ts,
    record,
  });
  if (error) throw error;
}

export function createSupabaseBackend(): AuditBackend {
  const supabase = createClient();
  return {
    mode: "supabase",
    async list(clipKey: string) {
      let online: AuditRecord[] | null = null;
      if (typeof navigator === "undefined" || navigator.onLine !== false) {
        try {
          const { data, error } = await supabase
            .from("sail_audits")
            .select("record")
            .eq("clip_key", clipKey)
            .order("created_at", { ascending: true });
          if (error) throw error;
          online = (data ?? [])
            .map((row) => AuditRecordSchema.safeParse(row.record))
            .filter((p) => p.success)
            .map((p) => p.data);
        } catch (e) {
          if (!isNetworkError(e)) throw e;
        }
      }
      if (online === null) {
        online = (await packedClip(clipKey))?.audits ?? [];
      }
      const queued = (await queuedAudits(clipKey)).map((q) => q.record);
      return mergeAudits(online, queued);
    },
    async append(record: AuditRecord) {
      if (typeof navigator !== "undefined" && navigator.onLine === false) {
        await enqueueAudit(record);
        return;
      }
      // An audit write must never be lost: retry transient failures (a
      // Supabase 429/5xx classifies "rejected", not "retry", but is just as
      // transient in practice), then queue whatever still fails — the queue
      // keeps rejected records with their error, shows in the pending pill,
      // and replays on click/reconnect. Duplicate key = an earlier attempt
      // actually landed (record.id is the PK): success.
      let lastErr: unknown = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          await insertSupabase(supabase, record);
          lastErr = null;
          break;
        } catch (e) {
          if (classifyError(e) === "duplicate") {
            lastErr = null;
            break;
          }
          lastErr = e;
          await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
        }
      }
      if (lastErr !== null) {
        await enqueueAudit(record);
        return;
      }
      // success: opportunistically drain records stranded by earlier
      // failures (the 'online' event never fires when we stayed online)
      void queuedCount().then((n) => {
        if (n > 0) void syncPendingAudits();
      });
    },
  };
}

let cached: AuditBackend | null = null;

export function getAuditBackend(): AuditBackend {
  if (!cached) {
    cached = isSupabaseConfigured()
      ? createSupabaseBackend()
      : createLocalBackend();
  }
  return cached;
}

/** Replay queued audits into Supabase. In demo mode nothing is ever
    queued (localStorage never fails), so this is a no-op there. */
export async function syncPendingAudits(): Promise<SyncReport> {
  if (!isSupabaseConfigured()) {
    return { synced: 0, duplicates: 0, rejected: 0, remaining: 0, stoppedOnNetwork: false };
  }
  const supabase = createClient();
  return syncQueue((record) => insertSupabase(supabase, record));
}

/** Latest verdict per frame_ts (append-only log -> current state). */
export function foldAudits(records: AuditRecord[]): Map<string, AuditRecord> {
  const byFrame = new Map<string, AuditRecord>();
  for (const rec of records) {
    const prev = byFrame.get(rec.frame_ts);
    if (!prev || rec.created_at >= prev.created_at) {
      byFrame.set(rec.frame_ts, rec);
    }
  }
  return byFrame;
}

/** Serialise folded audits to the repo's training-frames JSONL shape. */
export function auditsToJsonl(
  records: AuditRecord[],
  size: { width: number; height: number },
): string {
  const folded = [...foldAudits(records).values()].sort((a, b) =>
    a.frame_ts.localeCompare(b.frame_ts),
  );
  return folded
    .map((rec) =>
      JSON.stringify({
        frame_id: rec.frame_id,
        scene: rec.clip_key.split("__")[0],
        triplet_ts: rec.clip_key.split("__")[1],
        frame_ts: rec.frame_ts,
        source: "dashboard-web",
        audited: true,
        verdict: rec.verdict,
        fisheye_bboxes: rec.boxes.map(({ cls, xyxy, centroid }) =>
          centroid ? { cls, xyxy, centroid } : { cls, xyxy }),
        width: size.width,
        height: size.height,
      }),
    )
    .join("\n");
}
