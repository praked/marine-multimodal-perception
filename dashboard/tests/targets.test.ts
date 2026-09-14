/* Protocol v1.2 `targets`: schema round-trip + the pure drawing helpers.
   The wire shape mirrors target_to_dict in scripts/sensor_processing/
   target_motion.py (polar position, nullable velocity/CPA channels). */

import { SectorRecordSchema, TargetSchema, type Target } from "@/lib/types";
import {
  ARROW_MAX_M,
  ARROW_SCALE_S,
  TARGET_STATE_VAR,
  targetArrowEnd,
  targetCpaXY,
  targetLabel,
  targetXY,
} from "@/lib/viewer/targets";
import { describe, expect, it } from "vitest";

const fullTarget: Target = {
  id: 7,
  bearing_deg: -12,
  range_m: 4.2,
  n_points: 9,
  age_frames: 14,
  v_radial_mps: -0.61,
  v_tangential_mps: 0.11,
  vx_mps: 0.05,
  vy_mps: -0.6,
  speed_mps: 0.602,
  course_deg: 175.2,
  closing_mps: 0.61,
  cpa_m: 1.8,
  t_cpa_s: 12.4,
  motion_state: "closing",
  ego_corrected: true,
};

/** A track whose velocity channels have not warmed up yet. */
const warmupTarget: Target = {
  id: 8,
  bearing_deg: 30,
  range_m: 7.1,
  n_points: 3,
  age_frames: 3,
  v_radial_mps: null,
  v_tangential_mps: null,
  vx_mps: null,
  vy_mps: null,
  speed_mps: null,
  course_deg: null,
  closing_mps: null,
  cpa_m: null,
  t_cpa_s: null,
  motion_state: "unknown",
  ego_corrected: false,
};

function makeRecord(targets?: Target[]): Record<string, unknown> {
  const bins = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
  return {
    protocol: 1,
    timestamp: "16:37:02.5",
    clip_id: "2026-07-08/2026-07-08_16-37-01",
    bin_centers_deg: bins,
    scores: bins.map(() => 0),
    min_range_m: bins.map(() => null),
    sensor_hits: bins.map(() => [0, 0, 0]),
    tracked: false,
    ...(targets ? { targets } : {}),
  };
}

describe("TargetSchema (protocol v1.2)", () => {
  it("round-trips a fully-populated target through JSON unchanged", () => {
    const parsed = TargetSchema.parse(
      JSON.parse(JSON.stringify(fullTarget)),
    );
    expect(parsed).toEqual(fullTarget);
  });

  it("accepts null velocity + CPA channels (warm-up) and unknown state", () => {
    const parsed = TargetSchema.parse(warmupTarget);
    expect(parsed.speed_mps).toBeNull();
    expect(parsed.cpa_m).toBeNull();
    expect(parsed.motion_state).toBe("unknown");
  });

  it("rejects an unrecognised motion_state", () => {
    expect(() =>
      TargetSchema.parse({ ...fullTarget, motion_state: "orbiting" }),
    ).toThrow();
  });

  it("sector records carry targets optionally — absent stays absent", () => {
    const without = SectorRecordSchema.parse(makeRecord());
    expect(without.targets).toBeUndefined();
    const withTargets = SectorRecordSchema.parse(
      makeRecord([fullTarget, warmupTarget]),
    );
    expect(withTargets.targets).toHaveLength(2);
    expect(withTargets.targets![0]).toEqual(fullTarget);
  });
});

describe("target drawing helpers", () => {
  it("derives the boat-frame position from bearing + range", () => {
    const dead_ahead = targetXY({ ...fullTarget, bearing_deg: 0, range_m: 5 });
    expect(dead_ahead[0]).toBeCloseTo(0);
    expect(dead_ahead[1]).toBeCloseTo(5);
    const starboard = targetXY({ ...fullTarget, bearing_deg: 90, range_m: 5 });
    expect(starboard[0]).toBeCloseTo(5);
    expect(starboard[1]).toBeCloseTo(0);
  });

  it("scales the velocity arrow with speed and caps it", () => {
    const slow = { ...fullTarget, vx_mps: 0, vy_mps: 0.5, speed_mps: 0.5 };
    const [sx, sy] = targetXY(slow);
    const slowEnd = targetArrowEnd(slow)!;
    expect(Math.hypot(slowEnd[0] - sx, slowEnd[1] - sy)).toBeCloseTo(
      0.5 * ARROW_SCALE_S,
    );
    const fast = { ...fullTarget, vx_mps: 0, vy_mps: 10, speed_mps: 10 };
    const fastEnd = targetArrowEnd(fast)!;
    const [fx, fy] = targetXY(fast);
    expect(Math.hypot(fastEnd[0] - fx, fastEnd[1] - fy)).toBeCloseTo(
      ARROW_MAX_M,
    );
  });

  it("returns no arrow or CPA while the velocity is warming up", () => {
    expect(targetArrowEnd(warmupTarget)).toBeNull();
    expect(targetCpaXY(warmupTarget)).toBeNull();
  });

  it("predicts the CPA ghost at position + velocity × t_cpa", () => {
    const t = {
      ...fullTarget,
      bearing_deg: 0,
      range_m: 6,
      vx_mps: 0.2,
      vy_mps: -0.5,
      cpa_m: 1.5,
      t_cpa_s: 10,
    };
    const cpa = targetCpaXY(t)!;
    expect(cpa[0]).toBeCloseTo(0 + 0.2 * 10);
    expect(cpa[1]).toBeCloseTo(6 - 0.5 * 10);
  });

  it("maps every motion state to a theme token", () => {
    expect(TARGET_STATE_VAR.closing).toBe("var(--status-serious)");
    expect(TARGET_STATE_VAR.crossing).toBe("var(--viz-water-edge)");
    for (const v of Object.values(TARGET_STATE_VAR)) {
      expect(v).toMatch(/^var\(--/);
    }
  });

  it("describes a target for screen readers", () => {
    expect(targetLabel(fullTarget)).toBe(
      "target 7: closing, 4.2 m at -12°, closing 0.61 m/s, CPA 1.8 m in 12 s",
    );
    expect(targetLabel(warmupTarget)).toBe(
      "target 8: unknown, 7.1 m at 30°, closing unknown",
    );
  });
});
