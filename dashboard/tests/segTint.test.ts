import { SEG_PALETTE } from "@/lib/palette";
import {
  decodeSeg,
  edgeSegments,
  OBSTACLE,
  SKY,
  smoothEdge,
  tintOverlay,
  WATER,
  waterEdgeColumns,
} from "@/lib/viewer/segTint";
import { describe, expect, it } from "vitest";

function rgbaFromClasses(cls: number[], palette = false): Uint8ClampedArray {
  const out = new Uint8ClampedArray(cls.length * 4);
  cls.forEach((c, i) => {
    if (palette) {
      const [r, g, b] = SEG_PALETTE[c]!;
      out[i * 4] = r;
      out[i * 4 + 1] = g;
      out[i * 4 + 2] = b;
    } else {
      out[i * 4] = c;
      out[i * 4 + 1] = c;
      out[i * 4 + 2] = c;
    }
    out[i * 4 + 3] = 255;
  });
  return out;
}

describe("decodeSeg (mirrors segmentation.decode_seg_array)", () => {
  it("reads label-encoded grayscale masks", () => {
    const rgba = rgbaFromClasses([OBSTACLE, WATER, SKY, WATER]);
    expect([...decodeSeg(rgba, 2, 2)]).toEqual([0, 1, 2, 1]);
  });

  it("decodes eWaSR palette colours to nearest class", () => {
    const rgba = rgbaFromClasses([OBSTACLE, WATER, SKY], true);
    expect([...decodeSeg(rgba, 3, 1)]).toEqual([0, 1, 2]);
  });
});

describe("tintOverlay", () => {
  it("colours pixels by class at the requested alpha", () => {
    const cls = new Uint8Array([WATER]);
    const out = tintOverlay(cls, 1, 1, 0.4);
    expect([out[0], out[1], out[2]]).toEqual(SEG_PALETTE[WATER]);
    expect(out[3]).toBe(Math.round(0.4 * 255));
  });
});

describe("water edge", () => {
  // 3x3: column 0 water from row 1; column 1 no water; column 2 water at row 0
  const cls = new Uint8Array([
    SKY, SKY, WATER,
    WATER, SKY, WATER,
    WATER, OBSTACLE, WATER,
  ]);

  it("finds the topmost water row per column, -1 when absent", () => {
    expect([...waterEdgeColumns(cls, 3, 3)]).toEqual([1, -1, 0]);
  });

  it("median-smooths across missing columns", () => {
    const edge = new Int16Array([10, -1, 12]);
    const sm = smoothEdge(edge, 3);
    expect(sm[1]).toBeGreaterThanOrEqual(10);
    expect(sm[1]).toBeLessThanOrEqual(12);
  });

  it("splits segments on vertical jumps (poster: |Δy| >= 40)", () => {
    const edge = new Int16Array([100, 101, 102, 190, 191, 192]);
    const segs = edgeSegments(edge, 40);
    expect(segs).toHaveLength(2);
    expect(segs[0]!.map(([x]) => x)).toEqual([0, 1, 2]);
    expect(segs[1]!.map(([x]) => x)).toEqual([3, 4, 5]);
  });

  it("drops single-point segments", () => {
    const edge = new Int16Array([-1, 50, -1]);
    expect(edgeSegments(edge, 40)).toHaveLength(0);
  });
});
