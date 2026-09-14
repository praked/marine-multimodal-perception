import { describe, expect, it } from "vitest";
describe("degraded thermal health", () => {
  it("maps absent/degraded/ok frames to tri-state runs", async () => {
    const { thermalHealth, runsFromStates } = await import("@/lib/health");
    const frames = [
      { ts: "12:00:00.0", fisheye: true, thermal: true },
      { ts: "12:00:00.3", fisheye: true, thermal: true, thermal_q: "low" },
      { ts: "12:00:00.7", fisheye: true, thermal: false },
    ] as never[];
    const states = thermalHealth(frames);
    expect(states).toEqual(["ok", "degraded", "down"]);
    const runs = runsFromStates(states);
    expect(runs).toHaveLength(3);
    expect(runs.map((r) => r.state)).toEqual(["ok", "degraded", "down"]);
  });
});
