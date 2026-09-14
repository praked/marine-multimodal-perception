import { AuditRecordSchema, type AuditRecord } from "@/lib/types";
import { idbDelete, idbGetAll, idbPut, STORE_QUEUE } from "./db";

/* Offline audit queue. When the audit store is unreachable (no network,
   Supabase down) records land here instead of being lost; `syncQueue`
   replays them in order. The audit log is append-only and every record
   carries a client-side uuid (`id` = primary key in sail_audits), so a
   replay that races a previous partial success hits a duplicate-key error
   — which counts as synced, never as a second row. */

export interface QueuedAudit {
  id: string; // = record.id
  clip_key: string;
  record: AuditRecord;
  queued_at: string;
  attempts: number;
  last_error?: string;
}

export const QUEUE_EVENT = "asvproject-audit-queue";

function notify(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(QUEUE_EVENT));
}

export async function enqueueAudit(record: AuditRecord): Promise<void> {
  const item: QueuedAudit = {
    id: record.id,
    clip_key: record.clip_key,
    record,
    queued_at: new Date().toISOString(),
    attempts: 0,
  };
  await idbPut(STORE_QUEUE, item);
  notify();
}

export async function queuedAudits(clipKey?: string): Promise<QueuedAudit[]> {
  const all = (await idbGetAll<QueuedAudit>(STORE_QUEUE))
    .filter((q) => AuditRecordSchema.safeParse(q.record).success)
    .sort((a, b) => a.queued_at.localeCompare(b.queued_at));
  return clipKey ? all.filter((q) => q.clip_key === clipKey) : all;
}

export async function queuedCount(): Promise<number> {
  return (await idbGetAll<QueuedAudit>(STORE_QUEUE)).length;
}

export function subscribeQueue(cb: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(QUEUE_EVENT, cb);
  return () => window.removeEventListener(QUEUE_EVENT, cb);
}

/** Outcome of one replay attempt for a queued record. */
export type SyncOutcome = "synced" | "duplicate" | "retry" | "rejected";

/** Classify an insert failure: a duplicate key means the row already
    exists (synced), a network failure means try later, anything else is a
    server-side rejection (kept in the queue with the error for a human). */
export function classifyError(e: unknown): Exclude<SyncOutcome, "synced"> {
  const err = e as { code?: string; message?: string; status?: number } | null;
  const code = err?.code ?? "";
  const msg = String(err?.message ?? err ?? "");
  if (code === "23505" || /duplicate key/i.test(msg)) return "duplicate";
  if (isNetworkError(e)) return "retry";
  return "rejected";
}

export function isNetworkError(e: unknown): boolean {
  if (typeof navigator !== "undefined" && navigator.onLine === false) return true;
  const err = e as { message?: string; name?: string; status?: number } | null;
  const msg = String(err?.message ?? err ?? "");
  return (
    err?.name === "TypeError" ||
    /failed to fetch|networkerror|network request failed|load failed|fetch failed|ECONN|ENOTFOUND|timed? ?out/i.test(msg) ||
    err?.status === 0
  );
}

export interface SyncReport {
  synced: number;
  duplicates: number;
  rejected: number;
  remaining: number;
  stoppedOnNetwork: boolean;
}

/** Replay the queue through `append` (the online store's insert). Stops at
    the first network failure (order preserved); rejected records stay
    queued with their error so nothing silently vanishes. */
export async function syncQueue(
  append: (record: AuditRecord) => Promise<void>,
): Promise<SyncReport> {
  const report: SyncReport = { synced: 0, duplicates: 0, rejected: 0, remaining: 0, stoppedOnNetwork: false };
  const items = await queuedAudits();
  for (const item of items) {
    try {
      await append(item.record);
      await idbDelete(STORE_QUEUE, item.id);
      report.synced++;
    } catch (e) {
      const kind = classifyError(e);
      if (kind === "duplicate") {
        await idbDelete(STORE_QUEUE, item.id);
        report.duplicates++;
      } else if (kind === "retry") {
        report.stoppedOnNetwork = true;
        await idbPut(STORE_QUEUE, { ...item, attempts: item.attempts + 1, last_error: String((e as Error)?.message ?? e) });
        break;
      } else {
        report.rejected++;
        await idbPut(STORE_QUEUE, { ...item, attempts: item.attempts + 1, last_error: String((e as Error)?.message ?? e) });
      }
    }
  }
  report.remaining = await queuedCount();
  notify();
  return report;
}
