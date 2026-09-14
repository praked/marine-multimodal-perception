/* Pure helpers for drawing protocol-v1.2 tracked targets on the radar
   bird's-eye. Wire position is polar (bearing_deg right+, 0 = bow;
   range_m); the boat-frame x/y and the CPA ghost are derived here so the
   SVG layer stays declarative. Same data frame as lib/viewer/polar.ts. */

import type { MotionState, Target } from "@/lib/types";
import { polarPoint } from "@/lib/viewer/polar";

/** Seconds of travel one velocity arrow represents. */
export const ARROW_SCALE_S = 2.0;
/** Cap on the arrow length, metres (≈1.25 m/s and above saturate). */
export const ARROW_MAX_M = 2.5;

/** Motion-state colour tokens (theme vars, dataviz-validated palette):
    closing = the serious status tone, crossing = the reserved amber
    accent (the COLREG give-way case), static/diverging/unknown muted. */
export const TARGET_STATE_VAR: Record<MotionState, string> = {
  closing: "var(--status-serious)",
  crossing: "var(--viz-water-edge)",
  diverging: "var(--viz-trail)",
  static: "var(--subtle)",
  unknown: "var(--subtle)",
};

/** Boat-frame position [x lateral right+, y forward] in metres. */
export function targetXY(t: Target): [number, number] {
  return polarPoint(t.bearing_deg, t.range_m);
}

/** Boat-frame velocity [vx, vy] m/s, or null while the channel warms up. */
export function targetVelocityXY(t: Target): [number, number] | null {
  if (t.vx_mps == null || t.vy_mps == null) return null;
  return [t.vx_mps, t.vy_mps];
}

/** Endpoint of the velocity arrow: position + v̂ · min(speed·τ, cap).
    Null when the 2-D velocity is unknown or the speed is zero. */
export function targetArrowEnd(t: Target): [number, number] | null {
  const v = targetVelocityXY(t);
  if (v === null) return null;
  const speed = Math.hypot(v[0], v[1]);
  if (speed <= 0) return null;
  const len = Math.min(speed * ARROW_SCALE_S, ARROW_MAX_M);
  const [x, y] = targetXY(t);
  return [x + (v[0] / speed) * len, y + (v[1] / speed) * len];
}

/** Predicted closest-point-of-approach position (position + v · t_cpa),
    metres. Null when the CPA channels or the 2-D velocity are absent. */
export function targetCpaXY(t: Target): [number, number] | null {
  const v = targetVelocityXY(t);
  if (v === null || t.cpa_m == null || t.t_cpa_s == null) return null;
  const [x, y] = targetXY(t);
  return [x + v[0] * t.t_cpa_s, y + v[1] * t.t_cpa_s];
}

function fmt(v: number | null, digits: number, suffix: string): string {
  return v == null ? "unknown" : `${v.toFixed(digits)}${suffix}`;
}

/** Screen-reader / tooltip description of one target. */
export function targetLabel(t: Target): string {
  const parts = [
    `target ${t.id}: ${t.motion_state}`,
    `${t.range_m.toFixed(1)} m at ${Math.round(t.bearing_deg)}°`,
    `closing ${fmt(t.closing_mps, 2, " m/s")}`,
  ];
  if (t.cpa_m != null && t.t_cpa_s != null) {
    parts.push(`CPA ${t.cpa_m.toFixed(1)} m in ${t.t_cpa_s.toFixed(0)} s`);
  }
  return parts.join(", ");
}
