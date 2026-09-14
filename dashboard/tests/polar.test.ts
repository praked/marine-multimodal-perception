import {
  arcPath,
  polarPoint,
  RADAR_VIEW,
  SCORER_VIEW,
  sectorPath,
  spokePath,
  toPx,
} from "@/lib/viewer/polar";
import { describe, expect, it } from "vitest";

describe("polar geometry (mirrors render_hero.py)", () => {
  it("polarPoint: 0° points forward, +90° points starboard", () => {
    const [x0, y0] = polarPoint(0, 9);
    expect(x0).toBeCloseTo(0);
    expect(y0).toBeCloseTo(9);
    const [x90, y90] = polarPoint(90, 9);
    expect(x90).toBeCloseTo(9);
    expect(y90).toBeCloseTo(0);
  });

  it("toPx maps the data window onto the pixel viewport, y flipped", () => {
    const [ox, oy] = toPx(0, 0, RADAR_VIEW);
    expect(ox).toBeCloseTo(RADAR_VIEW.width / 2);
    expect(oy).toBeCloseTo(RADAR_VIEW.height); // origin at the bottom
    const [, topY] = toPx(0, RADAR_VIEW.yMax, RADAR_VIEW);
    expect(topY).toBeCloseTo(0);
  });

  it("spokePath starts at the origin", () => {
    const d = spokePath(30, 9, RADAR_VIEW);
    const [ox, oy] = toPx(0, 0, RADAR_VIEW);
    expect(d.startsWith(`M${ox.toFixed(2)},${oy.toFixed(2)}`)).toBe(true);
  });

  it("arcPath stays at constant radius", () => {
    const d = arcPath(9, -55, 55, SCORER_VIEW);
    const coords = [...d.matchAll(/([ML])([\d.]+),([\d.]+)/g)].map((m) => [
      Number(m[2]),
      Number(m[3]),
    ]);
    expect(coords.length).toBeGreaterThan(10);
    const [ox, oy] = toPx(0, 0, SCORER_VIEW);
    const sx = SCORER_VIEW.width / (SCORER_VIEW.xMax - SCORER_VIEW.xMin);
    const sy = SCORER_VIEW.height / (SCORER_VIEW.yMax - SCORER_VIEW.yMin);
    for (const [px, py] of coords) {
      const r = Math.hypot((px! - ox) / sx, (py! - oy) / sy);
      expect(r).toBeCloseTo(9, 1);
    }
  });

  it("sectorPath closes back at the origin and honours the angular gap", () => {
    const d = sectorPath(0, 5, 9, SCORER_VIEW, 0.9);
    expect(d.endsWith("Z")).toBe(true);
    // first arc point should sit at bearing -5 + 0.9 = -4.1°, not -5°
    const first = /M[\d.,]+ L([\d.]+),/.exec(d);
    expect(first).not.toBeNull();
    const gapless = sectorPath(0, 5, 9, SCORER_VIEW, 0);
    expect(d).not.toEqual(gapless);
  });
});
