import {
  angleToBin,
  binCenters,
  binEdges,
  DEFAULT_BINS,
  formatBearing,
  hitsFromScore,
  pointBearingDeg,
} from "@/lib/bins";
import { describe, expect, it } from "vitest";

describe("bin math (mirrors fusion.py make_bins)", () => {
  it("produces the canonical 11 ten-degree bins from -55..+55", () => {
    expect(binEdges(DEFAULT_BINS)).toHaveLength(12);
    expect(binCenters(DEFAULT_BINS)).toEqual([
      -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50,
    ]);
  });

  it("maps angles to bins with inclusive outer edges", () => {
    expect(angleToBin(-55, DEFAULT_BINS)).toBe(0);
    expect(angleToBin(0, DEFAULT_BINS)).toBe(5);
    expect(angleToBin(54.9, DEFAULT_BINS)).toBe(10);
    expect(angleToBin(55, DEFAULT_BINS)).toBe(10);
    expect(angleToBin(-55.1, DEFAULT_BINS)).toBeNull();
    expect(angleToBin(56, DEFAULT_BINS)).toBeNull();
  });

  it("formats bearings in the dashboard's +10°/−20°/0° style", () => {
    expect(formatBearing(10)).toBe("+10°");
    expect(formatBearing(-20)).toBe("−20°");
    expect(formatBearing(0)).toBe("0°");
  });

  it("computes bearing as atan2(X lateral, Y forward)", () => {
    expect(pointBearingDeg(0, 5)).toBeCloseTo(0);
    expect(pointBearingDeg(5, 5)).toBeCloseTo(45);
    expect(pointBearingDeg(-5, 5)).toBeCloseTo(-45);
  });

  it("recovers sensor-hit counts from n/3 scores", () => {
    expect(hitsFromScore(0)).toBe(0);
    expect(hitsFromScore(1 / 3)).toBe(1);
    expect(hitsFromScore(2 / 3)).toBe(2);
    expect(hitsFromScore(1)).toBe(3);
  });
});
