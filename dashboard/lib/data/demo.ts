import {
  CatalogueSchema,
  ClipMetaSchema,
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

const BASE = "/demo";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}/${path}`, { cache: "force-cache" });
  if (!res.ok) throw new Error(`demo asset ${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

/** Static-bundle provider: everything under /public/demo. */
export function createDemoProvider(): DataProvider {
  return {
    mode: "demo",
    async listClips(): Promise<ClipSummary[]> {
      const raw = await fetchJson<unknown>("clips.json");
      return CatalogueSchema.parse(raw).clips;
    },
    async getMeta(clipKey: string): Promise<ClipMeta> {
      const raw = await fetchJson<unknown>(`${clipKey}/meta.json`);
      return ClipMetaSchema.parse(raw);
    },
    getSectors(clipKey: string) {
      return fetchJson<Record<string, SectorRecord>>(`${clipKey}/sectors.json`);
    },
    getRadar(clipKey: string) {
      return fetchJson<RadarFrames>(`${clipKey}/radar.json`);
    },
    getBoxes(clipKey: string) {
      return fetchJson<Record<string, Box[]>>(`${clipKey}/boxes.json`);
    },
    getInstances(clipKey: string) {
      return fetchJson<Record<string, Instance[]>>(`${clipKey}/instances.json`);
    },
    getLabels(clipKey: string) {
      return fetchJson<Record<string, LabelRecord>>(`${clipKey}/labels.json`);
    },
    frameUrl(clipKey: string, kind: FrameKind, ts: string): string {
      const ext = kind === "seg" || kind === "thermal_hd" ? "png" : "jpg";
      return `${BASE}/${clipKey}/${kind}/ts=${safeTs(ts)}.${ext}`;
    },
    async resolveAssetUrl(clipKey: string, kind: FrameKind, ts: string) {
      return this.frameUrl(clipKey, kind, ts); // static files: no signing
    },
  };
}
