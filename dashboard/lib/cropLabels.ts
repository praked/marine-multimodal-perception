/* Write side of /crops: append-only crop labels in public.sail_crop_labels.

   A keypress must never be lost silently (same discipline as the audit
   backend): the insert retries 3x with backoff; a duplicate key means an
   earlier attempt landed (client uuid is the PK) and counts as success;
   whatever still fails goes to a localStorage queue that is drained on the
   next successful write (or an explicit retry). Without Supabase creds
   (CI, demo) labels persist to localStorage so the whole loop still works. */

import { isSupabaseConfigured } from "@/lib/data/provider";
import { createClient } from "@/lib/supabase/client";
import type { CropLabelRecord, CropVerdict } from "@/lib/crops";
import { foldCropLabels } from "@/lib/crops";

const LS_LOCAL = "asvproject.cropLabels.v1"; // demo-mode store
const LS_QUEUE = "asvproject.cropLabels.queue.v1"; // failed-write queue

function readLs(key: string): CropLabelRecord[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(window.localStorage.getItem(key) ?? "[]") as CropLabelRecord[];
  } catch {
    return [];
  }
}

function writeLs(key: string, records: CropLabelRecord[]) {
  window.localStorage.setItem(key, JSON.stringify(records));
}

export function makeRecord(cropId: string, frameId: string | undefined, label: CropVerdict): CropLabelRecord {
  return {
    id: crypto.randomUUID(),
    crop_id: cropId,
    frame_id: frameId,
    label,
    created_at: new Date().toISOString(),
  };
}

export function queuedCropLabels(): CropLabelRecord[] {
  return readLs(LS_QUEUE);
}

async function insertRow(rec: CropLabelRecord): Promise<void> {
  const supabase = createClient();
  const { error } = await supabase.from("sail_crop_labels").insert({
    id: rec.id,
    crop_id: rec.crop_id,
    frame_id: rec.frame_id ?? null,
    label: rec.label,
    created_at: rec.created_at,
  });
  if (error) {
    // 23505 = duplicate PK: an earlier attempt actually landed — success.
    if (error.code === "23505") return;
    throw error;
  }
}

/** Fetch every crop label, paged past Supabase's 1000-row cap; merges the
    still-queued records so the UI always shows what the labeller did. */
export async function listCropLabels(): Promise<CropLabelRecord[]> {
  if (!isSupabaseConfigured()) return readLs(LS_LOCAL);
  const supabase = createClient();
  const out: CropLabelRecord[] = [];
  const page = 1000;
  for (let from = 0; ; from += page) {
    const { data, error } = await supabase
      .from("sail_crop_labels")
      .select("id, crop_id, frame_id, label, created_at")
      .order("created_at", { ascending: true })
      .range(from, from + page - 1);
    if (error) throw error;
    out.push(...((data ?? []) as CropLabelRecord[]));
    if (!data || data.length < page) break;
  }
  const queued = queuedCropLabels();
  return queued.length ? [...out, ...queued] : out;
}

export type AppendOutcome = "saved" | "queued" | "local";

/** Append one label. Never throws; "queued" means the write failed after
    retries and sits in localStorage awaiting drainCropQueue(). */
export async function appendCropLabel(rec: CropLabelRecord): Promise<AppendOutcome> {
  if (!isSupabaseConfigured()) {
    writeLs(LS_LOCAL, [...readLs(LS_LOCAL), rec]);
    return "local";
  }
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await insertRow(rec);
      if (queuedCropLabels().length > 0) void drainCropQueue();
      return "saved";
    } catch {
      await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
    }
  }
  writeLs(LS_QUEUE, [...readLs(LS_QUEUE), rec]);
  return "queued";
}

/** Replay the failed-write queue; returns how many records remain queued. */
export async function drainCropQueue(): Promise<number> {
  if (!isSupabaseConfigured()) return 0;
  const queue = readLs(LS_QUEUE);
  const remaining: CropLabelRecord[] = [];
  for (const rec of queue) {
    try {
      await insertRow(rec);
    } catch {
      remaining.push(rec);
    }
  }
  writeLs(LS_QUEUE, remaining);
  return remaining.length;
}

export { foldCropLabels };
