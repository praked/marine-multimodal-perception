/* Angular-bin math, mirroring scripts/sensor_processing/fusion.py.
   Bearings: 0° = dead ahead (bow), positive = starboard. */

export interface BinConfig {
  minDeg: number;
  maxDeg: number;
  stepDeg: number;
}

export const DEFAULT_BINS: BinConfig = { minDeg: -55, maxDeg: 55, stepDeg: 10 };

export function binEdges({ minDeg, maxDeg, stepDeg }: BinConfig): number[] {
  const edges: number[] = [];
  for (let e = minDeg; e <= maxDeg + 1e-9; e += stepDeg) edges.push(e);
  return edges;
}

export function binCenters(cfg: BinConfig): number[] {
  const edges = binEdges(cfg);
  const centers: number[] = [];
  for (let i = 0; i < edges.length - 1; i++) {
    centers.push((edges[i]! + edges[i + 1]!) / 2);
  }
  return centers;
}

/** Index of the bin containing `angleDeg`, or null when outside all bins. */
export function angleToBin(angleDeg: number, cfg: BinConfig): number | null {
  if (angleDeg < cfg.minDeg || angleDeg > cfg.maxDeg) return null;
  const idx = Math.floor((angleDeg - cfg.minDeg) / cfg.stepDeg);
  const n = Math.round((cfg.maxDeg - cfg.minDeg) / cfg.stepDeg);
  return Math.min(idx, n - 1);
}

/** "+10°" / "−20°" / "0°" — matches the Tkinter dashboard's tick style. */
export function formatBearing(deg: number): string {
  if (deg === 0) return "0°";
  return `${deg > 0 ? "+" : "−"}${Math.abs(deg)}°`;
}

/** Bearing of a radar point: atan2(X lateral, Y forward), degrees. */
export function pointBearingDeg(x: number, y: number): number {
  return (Math.atan2(x, y) * 180) / Math.PI;
}

/** Score -> number of sensors that hit (score = nHits / 3). */
export function hitsFromScore(score: number, numSensors = 3): number {
  return Math.round(score * numSensors);
}
