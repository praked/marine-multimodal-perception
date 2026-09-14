import {
  hitTest,
  isValidBox,
  moveBox,
  normaliseBox,
  resizeBox,
} from "@/lib/annotate/boxes";
import type { Box } from "@/lib/types";
import { describe, expect, it } from "vitest";

const box = (xyxy: [number, number, number, number], cls = "boat"): Box => ({
  cls,
  xyxy,
});

describe("annotation box geometry", () => {
  it("normalises inverted corners", () => {
    expect(normaliseBox([0.8, 0.7, 0.2, 0.1])).toEqual([0.2, 0.1, 0.8, 0.7]);
  });

  it("hit-tests corners before bodies, topmost box first", () => {
    const boxes = [box([0.1, 0.1, 0.5, 0.5]), box([0.4, 0.4, 0.9, 0.9])];
    // corner of the lower box wins over the body of the upper box
    expect(hitTest(boxes, 0.5, 0.5, 0.02)).toEqual({ index: 0, corner: "br" });
    // body hit picks the topmost (last) box
    expect(hitTest(boxes, 0.45, 0.45, 0.001)).toEqual({
      index: 1,
      corner: null,
    });
    expect(hitTest(boxes, 0.95, 0.05, 0.02)).toBeNull();
  });

  it("moves boxes with clamping at the frame edge", () => {
    const moved = moveBox([0.1, 0.1, 0.3, 0.3], 0.2, 0);
    expect(moved[0]).toBeCloseTo(0.3);
    expect(moved[1]).toBeCloseTo(0.1);
    expect(moved[2]).toBeCloseTo(0.5);
    expect(moved[3]).toBeCloseTo(0.3);
    const clamped = moveBox([0.7, 0.7, 0.9, 0.9], 0.5, 0.5);
    expect(clamped[2]).toBeCloseTo(1);
    expect(clamped[3]).toBeCloseTo(1);
    expect(clamped[2]! - clamped[0]!).toBeCloseTo(0.2);
  });

  it("resizes by corner and renormalises when dragged across", () => {
    expect(resizeBox([0.2, 0.2, 0.6, 0.6], "br", 0.8, 0.9)).toEqual([
      0.2, 0.2, 0.8, 0.9,
    ]);
    // dragging tl past br flips the box rather than producing negatives
    const flipped = resizeBox([0.2, 0.2, 0.6, 0.6], "tl", 0.7, 0.7);
    expect(flipped).toEqual([0.6, 0.6, 0.7, 0.7]);
  });

  it("rejects sub-5px boxes like the label tool", () => {
    const minW = 5 / 864;
    const minH = 5 / 648;
    expect(isValidBox([0.1, 0.1, 0.1 + minW, 0.1 + minH], minW, minH)).toBe(
      true,
    );
    expect(isValidBox([0.1, 0.1, 0.102, 0.1005], minW, minH)).toBe(false);
  });
});
