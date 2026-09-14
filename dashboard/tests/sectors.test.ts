import { runsFromBooleans, fisheyeHealth, radarHealth } from "@/lib/health";
import { sectorEdges, sectorView } from "@/lib/sectors";
import type { FrameEntry, SectorRecord } from "@/lib/types";
import { describe, expect, it } from "vitest";

const rec: SectorRecord = {
  protocol: 1,
  timestamp: "16:00:00.0",
  clip_id: "x/y",
  bin_centers_deg: [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
  scores: [0.33, 0, 0, 0.66, 0, 1, 0, 0, 0.33, 0, 0],
  min_range_m: [1.5, null, null, 3, null, 0.8, null, null, null, null, null],
  sensor_hits: [
    [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1],
    [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
  ],
  tracked: false,
  confirmed: [false, false, false, true, false, false, false, false, false, false, false],
};

describe("sectorView (display-level bin resampling)", () => {
  it("width = native step is the identity (10° legacy record)", () => {
    const v = sectorView(rec, 10);
    expect(v.centers).toEqual(rec.bin_centers_deg);
    expect(v.scores).toEqual(rec.scores);
    expect(v.halfWidths.every((h) => h === 5)).toBe(true);
  });

  it("width = native step is the identity (15° D.2 record)", () => {
    const rec15: SectorRecord = {
      ...rec,
      bin_centers_deg: [-45, -30, -15, 0, 15, 30, 45],
      scores: [0, 0.33, 0, 1, 0, 0, 0.66],
      min_range_m: [null, 2, null, 0.8, null, null, 5],
      sensor_hits: rec.sensor_hits.slice(0, 7),
      confirmed: undefined,
    };
    const v = sectorView(rec15, 15);
    expect(v.centers).toEqual(rec15.bin_centers_deg);
    expect(v.scores).toEqual(rec15.scores);
    expect(v.halfWidths.every((h) => h === 7.5)).toBe(true);
  });

  it("resampling is conservative: max score, min range, any confirmed, OR hits", () => {
    const v = sectorView(rec, 30);
    // Grid centered on 0°: clipped edge sectors + three full 30° sectors.
    expect(v.centers).toEqual([-50, -30, 0, 30, 50]);
    expect(v.halfWidths).toEqual([5, 15, 15, 15, 5]);
    expect(v.scores[1]).toBe(0.66); // -40/-30/-20 bins -> max
    expect(v.scores[2]).toBe(1); // contains the 0° bin's 1.0
    expect(v.minRange[2]).toBe(0.8);
    expect(v.confirmed![1]).toBe(true); // -20° confirmed folds in
    expect(v.sensorHits[2]).toEqual([1, 1, 1]);
  });

  it("sub-native widths subdivide a bin without changing its value", () => {
    const v = sectorView(rec, 5);
    // The 0° native bin [-5, 5] shows up in both 5°-wide sectors it spans.
    const zeroish = v.centers
      .map((c, i) => ({ c, s: v.scores[i]! }))
      .filter(({ c }) => Math.abs(c) < 5);
    expect(zeroish.length).toBeGreaterThan(0);
    expect(zeroish.every(({ s }) => s === 1)).toBe(true);
  });

  it("sectorEdges spans -55..55 at every width", () => {
    for (const w of [5, 10, 15, 20, 25, 30, 35, 40, 45]) {
      const edges = sectorEdges(rec, w);
      expect(edges[0]).toBe(-55);
      expect(edges[edges.length - 1]).toBe(55);
    }
  });
});

describe("health timelines", () => {
  const mk = (ts: string, thermal = true): FrameEntry => ({
    ts, fisheye: true, thermal, seg: false,
  });

  it("compresses booleans into runs", () => {
    expect(runsFromBooleans([true, true, false, true])).toEqual([
      { start: 0, width: 0.5, state: "ok" },
      { start: 0.5, width: 0.25, state: "down" },
      { start: 0.75, width: 0.25, state: "ok" },
    ]);
  });

  it("fisheye health flags loop-rate gaps", () => {
    const frames = [mk("16:00:00.0"), mk("16:00:00.3"), mk("16:00:10.3"), mk("16:00:10.6")];
    expect(fisheyeHealth(frames)).toEqual([true, false, true, true]);
  });

  it("radar health tracks group presence near each frame", () => {
    const frames = [mk("16:00:00.0"), mk("16:00:01.0"), mk("16:00:30.0")];
    const radar = { "16:00:00.1": [[0, 1, 0]], "16:00:01.0": [[0, 2, 0]] };
    expect(radarHealth(frames, radar)).toEqual([false, true, false]);
  });
});
