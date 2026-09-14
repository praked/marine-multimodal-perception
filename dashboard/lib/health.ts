import type { FrameEntry, RadarFrames } from "@/lib/types";

/* Per-sensor health timelines for a clip: a boolean per frame, compressed
   into runs for rendering. Pure + tested. */

export type HealthState = "ok" | "degraded" | "down";

export interface HealthRun {
  start: number; // fraction 0..1 of the timeline
  width: number;
  state: HealthState;
}

export function runsFromStates(states: HealthState[]): HealthRun[] {
  const runs: HealthRun[] = [];
  if (states.length === 0) return runs;
  let runStart = 0;
  let state = states[0]!;
  for (let i = 1; i <= states.length; i++) {
    const v = i < states.length ? states[i]! : "__end__";
    if (v !== state) {
      runs.push({ start: runStart / states.length,
                  width: (i - runStart) / states.length, state });
      runStart = i;
      state = v as HealthState;
    }
  }
  return runs;
}

export function runsFromBooleans(ok: boolean[]): HealthRun[] {
  return runsFromStates(ok.map((v) => (v ? "ok" : "down")));
}

function tsSeconds(ts: string): number {
  const [h, m, s] = ts.split(":");
  return Number(h) * 3600 + Number(m) * 60 + Number(s);
}

/** Fisheye health = loop-rate continuity: a frame is unhealthy when the gap
    to the next frame exceeds `maxGapS` (nominal cadence is ~0.33 s; the
    2026-08-19 collapse ran at ~10 s/frame). Chunk-roll gaps (~10-20 s)
    also show — honestly — as brief drops. */
export function fisheyeHealth(frames: FrameEntry[], maxGapS = 2.5): boolean[] {
  return frames.map((f, i) => {
    if (i === frames.length - 1) return true;
    let delta = tsSeconds(frames[i + 1]!.ts) - tsSeconds(f.ts);
    if (delta < 0) delta += 86400; // midnight wrap
    return delta <= maxGapS;
  });
}

/** Tri-state: absent frame = down; present but below the quality-guard
    contrast floors (misted cover, day-1) = degraded; else ok. */
export function thermalHealth(frames: FrameEntry[]): HealthState[] {
  return frames.map((f) =>
    !f.thermal ? "down" : f.thermal_q === "low" ? "degraded" : "ok");
}

/** Radar health = a radar group exists at (or within `toleranceS` before)
    the frame's timestamp. Zero-return open water without heartbeat rows is
    indistinguishable from a dropout in the bundle — the tooltip says so. */
export function radarHealth(
  frames: FrameEntry[],
  radar: RadarFrames,
  toleranceS = 2.0,
): boolean[] {
  const keys = Object.keys(radar).sort();
  const secs = keys.map(tsSeconds);
  return frames.map((f) => {
    const t = tsSeconds(f.ts);
    // binary search nearest radar group at or before t
    let lo = 0;
    let hi = secs.length - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (secs[mid]! <= t + 0.05) {
        best = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return best >= 0 && t - secs[best]! <= toleranceS;
  });
}
