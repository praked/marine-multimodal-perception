import { isSupabaseConfigured } from "@/lib/data/provider";
import { createClient } from "@/lib/supabase/client";
import { normaliseCuts } from "./logic";
import {
  CurationRecordSchema,
  type Cut,
  type CurationAction,
  type CurationRecord,
} from "./types";

/* Curation store: one row per clip key (public.sail_curation) + an
   append-only history (sail_curation_log). Demo mode keeps the same shape
   in localStorage so the whole flow works with no credentials. Same seam
   as lib/audit/backend and lib/goals. */

export interface CurationBackend {
  readonly mode: "local" | "supabase";
  list(): Promise<Map<string, CurationRecord>>;
  setDeleted(clipKey: string, deleted: boolean, by?: string): Promise<CurationRecord>;
  setCuts(clipKey: string, cuts: Cut[], by?: string): Promise<CurationRecord>;
}

const LS_KEY = "asvproject.curation.v1";
export const DEFAULT_ACTOR = "dashboard-web";

function blank(clipKey: string): CurationRecord {
  return {
    clip_key: clipKey,
    deleted_at: null,
    restored_at: null,
    purged_at: null,
    cuts: [],
    note: null,
    updated_at: new Date(0).toISOString(),
    updated_by: null,
  };
}

/** Pure record transitions (tested): the backends only persist them. */
export function applyDeleted(rec: CurationRecord, deleted: boolean, nowIso: string, by: string): CurationRecord {
  return deleted
    ? { ...rec, deleted_at: nowIso, updated_at: nowIso, updated_by: by }
    : { ...rec, restored_at: nowIso, updated_at: nowIso, updated_by: by };
}

export function applyCuts(rec: CurationRecord, cuts: Cut[], nowIso: string, by: string): CurationRecord {
  return { ...rec, cuts: normaliseCuts(cuts), updated_at: nowIso, updated_by: by };
}

function readLocal(): Map<string, CurationRecord> {
  const out = new Map<string, CurationRecord>();
  if (typeof window === "undefined") return out;
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return out;
    for (const r of JSON.parse(raw) as unknown[]) {
      const p = CurationRecordSchema.safeParse(r);
      if (p.success) out.set(p.data.clip_key, p.data);
    }
  } catch {
    /* unreadable store: treat as empty */
  }
  return out;
}

function writeLocal(map: Map<string, CurationRecord>): void {
  window.localStorage.setItem(LS_KEY, JSON.stringify([...map.values()]));
}

export function createLocalCurationBackend(): CurationBackend {
  return {
    mode: "local",
    async list() {
      return readLocal();
    },
    async setDeleted(clipKey, deleted, by = DEFAULT_ACTOR) {
      const map = readLocal();
      const next = applyDeleted(map.get(clipKey) ?? blank(clipKey), deleted, new Date().toISOString(), by);
      map.set(clipKey, next);
      writeLocal(map);
      return next;
    },
    async setCuts(clipKey, cuts, by = DEFAULT_ACTOR) {
      const map = readLocal();
      const next = applyCuts(map.get(clipKey) ?? blank(clipKey), cuts, new Date().toISOString(), by);
      map.set(clipKey, next);
      writeLocal(map);
      return next;
    },
  };
}

export function createSupabaseCurationBackend(): CurationBackend {
  const supabase = createClient();

  async function current(clipKey: string): Promise<CurationRecord> {
    const { data, error } = await supabase
      .from("sail_curation")
      .select("*")
      .eq("clip_key", clipKey)
      .maybeSingle();
    if (error) throw error;
    if (!data) return blank(clipKey);
    return CurationRecordSchema.parse(data);
  }

  async function persist(rec: CurationRecord, action: CurationAction, by: string): Promise<CurationRecord> {
    if (rec.purged_at) {
      throw new Error("this set was purged: its bundle no longer exists");
    }
    const { error } = await supabase.from("sail_curation").upsert({
      clip_key: rec.clip_key,
      deleted_at: rec.deleted_at,
      restored_at: rec.restored_at,
      cuts: rec.cuts,
      note: rec.note ?? null,
      updated_at: rec.updated_at,
      updated_by: rec.updated_by ?? by,
    });
    if (error) throw error;
    const { error: logError } = await supabase.from("sail_curation_log").insert({
      clip_key: rec.clip_key,
      action,
      cuts: rec.cuts,
      note: rec.note ?? null,
      created_by: by,
    });
    if (logError) throw logError;
    return rec;
  }

  return {
    mode: "supabase",
    async list() {
      const { data, error } = await supabase.from("sail_curation").select("*");
      if (error) throw error;
      const out = new Map<string, CurationRecord>();
      for (const row of data ?? []) {
        const p = CurationRecordSchema.safeParse(row);
        if (p.success) out.set(p.data.clip_key, p.data);
      }
      return out;
    },
    async setDeleted(clipKey, deleted, by = DEFAULT_ACTOR) {
      const next = applyDeleted(await current(clipKey), deleted, new Date().toISOString(), by);
      return persist(next, deleted ? "delete" : "restore", by);
    },
    async setCuts(clipKey, cuts, by = DEFAULT_ACTOR) {
      const next = applyCuts(await current(clipKey), cuts, new Date().toISOString(), by);
      return persist(next, "set_cuts", by);
    },
  };
}

let cached: CurationBackend | null = null;

export function getCurationBackend(): CurationBackend {
  if (!cached) {
    cached = isSupabaseConfigured()
      ? createSupabaseCurationBackend()
      : createLocalCurationBackend();
  }
  return cached;
}

/** Records, or an empty map when the store is unreachable (offline, table
    missing): the catalogue must still render — a set then simply shows
    uncurated, which is the honest fallback. */
export async function loadCurationSafe(): Promise<Map<string, CurationRecord>> {
  try {
    return await getCurationBackend().list();
  } catch (e) {
    console.warn("[curation] store unreachable, showing uncurated catalogue:", e);
    return new Map();
  }
}
