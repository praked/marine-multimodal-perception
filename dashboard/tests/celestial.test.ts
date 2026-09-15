import {
  horizontalToVector,
  moonPosition,
  sunPosition,
} from "@/lib/celestial";
import { describe, expect, it } from "vitest";

const LAT = 46.000000;
const LON = 9.000000;

describe("sunPosition (mirrors the Python enrichment)", () => {
  it("matches the enrichment's validated July noon value", () => {
    const p = sunPosition(new Date("2026-07-08T11:20:00Z"), LAT, LON);
    expect(p.elevationDeg).toBeCloseTo(66.4, 0);
    expect(p.azimuthDeg).toBeGreaterThan(160);
    expect(p.azimuthDeg).toBeLessThan(195);
  });

  it("is below the horizon at local midnight", () => {
    const p = sunPosition(new Date("2026-07-08T23:00:00Z"), LAT, LON);
    expect(p.elevationDeg).toBeLessThan(-10);
  });
});

describe("moonPosition (truncated Meeus)", () => {
  it("stays in physical bounds and moves over hours", () => {
    const a = moonPosition(new Date("2026-08-19T14:00:00Z"), LAT, LON);
    const b = moonPosition(new Date("2026-08-19T20:00:00Z"), LAT, LON);
    for (const p of [a, b]) {
      expect(p.elevationDeg).toBeGreaterThan(-90);
      expect(p.elevationDeg).toBeLessThan(90);
      expect(p.azimuthDeg).toBeGreaterThanOrEqual(0);
      expect(p.azimuthDeg).toBeLessThan(360);
    }
    expect(Math.abs(a.azimuthDeg - b.azimuthDeg)).toBeGreaterThan(10);
  });
});

describe("horizontalToVector (scene frame: X east, Y up, Z south)", () => {
  it("maps the cardinal directions correctly", () => {
    const north = horizontalToVector({ elevationDeg: 0, azimuthDeg: 0 });
    expect(north[2]).toBeCloseTo(-1);
    const east = horizontalToVector({ elevationDeg: 0, azimuthDeg: 90 });
    expect(east[0]).toBeCloseTo(1);
    const up = horizontalToVector({ elevationDeg: 90, azimuthDeg: 123 });
    expect(up[1]).toBeCloseTo(1);
  });
});
