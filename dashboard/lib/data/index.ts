import { createDemoProvider } from "./demo";
import { isSupabaseConfigured, type DataProvider } from "./provider";
import { createSupabaseProvider } from "./supabaseProvider";

let cached: DataProvider | null = null;

/** The app-wide provider: Supabase when configured, demo bundle otherwise. */
export function getProvider(): DataProvider {
  if (!cached) {
    cached = isSupabaseConfigured()
      ? createSupabaseProvider()
      : createDemoProvider();
  }
  return cached;
}

export { isSupabaseConfigured } from "./provider";
export type { DataProvider, FrameKind } from "./provider";

import { applyCuration, type DeletedClip } from "@/lib/curation/logic";
import { loadCurationSafe } from "@/lib/curation/backend";
import type { CurationRecord } from "@/lib/curation/types";
import type { ClipSummary } from "@/lib/types";

export interface CuratedCatalogue {
  /** Sets that are not deleted (the default catalogue everywhere). */
  active: ClipSummary[];
  /** Soft-deleted sets with their records (the "Deleted (N)" view). */
  deleted: DeletedClip[];
  /** Every curation record by clip key (cuts for active sets live here). */
  records: Map<string, CurationRecord>;
}

/** The catalogue after curation: what every page should list by default.
    Curation-store failures degrade to the uncurated catalogue (logged). */
export async function listCuratedClips(): Promise<CuratedCatalogue> {
  const [clips, records] = await Promise.all([
    getProvider().listClips(),
    loadCurationSafe(),
  ]);
  const { active, deleted } = applyCuration(clips, records);
  return { active, deleted, records };
}
