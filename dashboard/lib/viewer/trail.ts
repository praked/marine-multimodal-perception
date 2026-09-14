/* Radar trail selection with an age gate. The radar can die mid-clip (e.g.
   the 2026-08-26 outing, 16:42→16:56): without a gate the viewer renders the
   last trail forever — frozen dots while the cameras play. A group only
   belongs to the trail if it is within MAX_TRAIL_AGE_S of the current frame. */

export const MAX_TRAIL_AGE_S = 2.0;

export function tsSeconds(ts: string): number {
  const [h, m, s] = ts.replace(/-/g, ":").split(":");
  return Number(h) * 3600 + Number(m) * 60 + Number(s);
}

export interface TrailResult {
  /** groups oldest→newest; empty when the radar is silent at this frame */
  groups: string[];
  /** newest radar timestamp ≤ the frame ts (null = none at all), for the
      "radar silent since …" note when the gate empties the trail */
  lastKey: string | null;
}

export function selectTrail(
  radarKeys: string[], // time-sorted RoundedTime keys
  ts: string,
  maxGroups: number,
  maxAgeS: number = MAX_TRAIL_AGE_S,
): TrailResult {
  const now = tsSeconds(ts);
  const pos = radarKeys.filter((k) => k <= ts);
  const lastKey = pos.length ? pos[pos.length - 1]! : null;
  const fresh = pos.filter((k) => now - tsSeconds(k) <= maxAgeS);
  return { groups: fresh.slice(-maxGroups), lastKey };
}
