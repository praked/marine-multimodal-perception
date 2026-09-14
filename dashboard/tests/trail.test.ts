import { describe, expect, it } from "vitest";
import { selectTrail, tsSeconds } from "@/lib/viewer/trail";

const keys = ["16:42:00.1", "16:42:00.4", "16:42:00.8", "16:42:01.1"];

describe("radar trail age gate", () => {
  it("returns the fresh window while the radar is alive", () => {
    const r = selectTrail(keys, "16:42:01.2", 3);
    expect(r.groups).toEqual(["16:42:00.4", "16:42:00.8", "16:42:01.1"]);
    expect(r.lastKey).toBe("16:42:01.1");
  });
  it("empties the trail when the radar has been silent past the gate", () => {
    const r = selectTrail(keys, "16:44:30.0", 8);
    expect(r.groups).toEqual([]);           // no frozen last trail
    expect(r.lastKey).toBe("16:42:01.1");   // for the "silent since" note
  });
  it("handles no radar at all", () => {
    const r = selectTrail([], "16:42:01.2", 8);
    expect(r.groups).toEqual([]);
    expect(r.lastKey).toBeNull();
  });
  it("tsSeconds handles dash and colon forms", () => {
    expect(tsSeconds("16-42-00.5")).toBeCloseTo(tsSeconds("16:42:00.5"));
  });
});
