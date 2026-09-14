import type {
  Box,
  ClipMeta,
  ClipSummary,
  Instance,
  LabelRecord,
  RadarFrames,
  SectorRecord,
} from "@/lib/types";

export type FrameKind = "frames" | "frames_hd" | "thermal" | "thermal_hd" | "seg";

/** Read side of the dashboard. Two implementations:
 *  - demo: static bundle under /public/demo (no credentials, CI, previews)
 *  - supabase: catalogue rows in Postgres, assets in Storage
 *  Both serve byte-identical bundle shapes (dashboard/tools/bake_demo.py). */
export interface DataProvider {
  readonly mode: "demo" | "supabase";
  listClips(): Promise<ClipSummary[]>;
  getMeta(clipKey: string): Promise<ClipMeta>;
  getSectors(clipKey: string): Promise<Record<string, SectorRecord>>;
  getRadar(clipKey: string): Promise<RadarFrames>;
  getBoxes(clipKey: string): Promise<Record<string, Box[]>>;
  getInstances(clipKey: string): Promise<Record<string, Instance[]>>;
  getLabels(clipKey: string): Promise<Record<string, LabelRecord>>;
  /** URL usable in a plain <img> for one frame asset (may redirect). */
  frameUrl(clipKey: string, kind: FrameKind, ts: string): string;
  /** Directly-loadable URL for CORS consumers (canvas pixels, fetch):
      resolves the signed R2 URL via the route's json mode so the browser
      never follows a cross-origin redirect (whose Origin serialises to
      "null" and fails the bucket's CORS policy). */
  resolveAssetUrl(clipKey: string, kind: FrameKind, ts: string): Promise<string>;
}

export function isSupabaseConfigured(): boolean {
  return Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL &&
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  );
}
