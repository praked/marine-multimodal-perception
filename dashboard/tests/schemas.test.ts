/* Contract tests against the REAL baked bundle: if bake_demo.py and the
   TypeScript schemas drift apart, these fail. */

import {
  BoxSchema,
  CatalogueSchema,
  ClipMetaSchema,
  InstanceSchema,
  LabelRecordSchema,
  SectorRecordSchema,
} from "@/lib/types";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const DEMO = path.resolve(__dirname, "..", "public", "demo");
const QUAD = "2026-07-08__2026-07-08_16-37-01";
const DAY1 = "2026-06-17_institutionone_day1__2026-06-17_13-16-25";

function loadJson<T>(...parts: string[]): T {
  return JSON.parse(readFileSync(path.join(DEMO, ...parts), "utf8")) as T;
}

describe("baked bundle matches the TypeScript contracts", () => {
  it("clips.json is a valid catalogue with both demo clips", () => {
    const catalogue = CatalogueSchema.parse(loadJson("clips.json"));
    const ids = catalogue.clips.map((c) => c.clip_id);
    expect(ids).toContain("2026-07-08/2026-07-08_16-37-01");
    expect(ids).toContain("2026-06-17_institutionone_day1/2026-06-17_13-16-25");
  });

  it("quad clip meta parses and indexes real frames", () => {
    const meta = ClipMetaSchema.parse(loadJson(QUAD, "meta.json"));
    expect(meta.streams).toMatchObject({
      fisheye: true,
      thermal: true,
      radar: true,
      sectors: true,
    });
    expect(meta.frames.length).toBe(meta.n_frames);
    expect(meta.frames[0]!.ts).toMatch(/^\d{2}:\d{2}:\d{2}\.\d$/);
  });

  it("every baked sector record parses (protocol v1.1 + scorer)", () => {
    const sectors = loadJson<Record<string, unknown>>(QUAD, "sectors.json");
    const entries = Object.entries(sectors);
    expect(entries.length).toBeGreaterThan(50);
    for (const [ts, raw] of entries) {
      const rec = SectorRecordSchema.parse(raw);
      expect(rec.timestamp).toBe(ts);
      expect(rec.bin_centers_deg).toHaveLength(11);
      expect(rec.scores).toHaveLength(11);
      expect(rec.sensor_hits).toHaveLength(11);
    }
    // the quad clip ran with --assoc and --scorer: those fields must survive
    const one = SectorRecordSchema.parse(entries[0]![1]);
    expect(one.confirmed).toBeDefined();
    expect(one.p_obstacle).toBeDefined();
    expect(one.threat).toBeDefined();
  });

  it("radar groups are numeric [x,y,z,v] rows", () => {
    const radar = loadJson<Record<string, number[][]>>(QUAD, "radar.json");
    const groups = Object.values(radar);
    expect(groups.length).toBeGreaterThan(500);
    for (const row of groups[0]!) {
      expect(row.length).toBeGreaterThanOrEqual(3);
      for (const v of row) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("boxes and labels parse for both clips", () => {
    for (const clip of [QUAD, DAY1]) {
      const boxes = loadJson<Record<string, unknown[]>>(clip, "boxes.json");
      for (const list of Object.values(boxes)) {
        for (const b of list) BoxSchema.parse(b);
      }
      const labels = loadJson<Record<string, unknown>>(clip, "labels.json");
      for (const rec of Object.values(labels)) LabelRecordSchema.parse(rec);
    }
  });

  it("day-1 instance polygons parse and are normalised", () => {
    const instances = loadJson<Record<string, unknown[]>>(
      DAY1,
      "instances.json",
    );
    expect(Object.keys(instances).length).toBeGreaterThan(10);
    for (const list of Object.values(instances)) {
      for (const raw of list) {
        const inst = InstanceSchema.parse(raw);
        expect(inst.polygon.length % 2).toBe(0);
        for (const v of inst.polygon) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1.0001);
        }
      }
    }
  });
});
