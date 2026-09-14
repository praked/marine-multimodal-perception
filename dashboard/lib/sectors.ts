import type { SectorRecord } from "@/lib/types";

/* Display-level sector resampling: regrid the record's native bins (10°
   legacy, 15° since the D.2 ruling — read per record from
   bin_centers_deg) onto a grid of `widthDeg`-wide sectors centered on
   dead-ahead. Each sector takes the CONSERVATIVE aggregate of every
   native bin it overlaps (max score/p/threat, min range, any confirmed,
   OR of sensor hits) — a resampled sector can only look more blocked,
   never safer. A width below the native bin just subdivides it (no fake
   resolution is created: neighbours repeat the bin's value). */

export interface SectorView {
  centers: number[];
  halfWidths: number[];
  scores: number[];
  minRange: (number | null)[];
  confirmed: boolean[] | undefined;
  pObstacle: number[] | undefined;
  threat: number[] | undefined;
  sensorHits: [number, number, number][];
  velocity: (number | null)[] | undefined;
  ttc: (number | null)[] | undefined;
}

const maxOf = (xs: number[]) => Math.max(...xs);
const minNullable = (xs: (number | null)[]): number | null => {
  const vals = xs.filter((v): v is number => v != null);
  return vals.length ? Math.min(...vals) : null;
};

export function sectorView(record: SectorRecord, widthDeg: number): SectorView {
  const binCenters = record.bin_centers_deg;
  const step = binCenters.length > 1 ? binCenters[1]! - binCenters[0]! : 15;
  const lo = binCenters[0]! - step / 2;
  const hi = binCenters[binCenters.length - 1]! + step / 2;
  const w = Math.min(Math.max(5, widthDeg), hi - lo);
  // Sector grid centered on 0° (dead ahead); the outermost sectors clip to
  // the record's span. A native bin belongs to every sector it overlaps
  // (open overlap: shared boundaries don't double-count).
  const EPS = 1e-6;
  const maxK = Math.ceil((hi - w / 2) / w);
  const sectors: { lo: number; hi: number; ii: number[] }[] = [];
  for (let k = -maxK; k <= maxK; k++) {
    const sLo = Math.max(lo, k * w - w / 2);
    const sHi = Math.min(hi, k * w + w / 2);
    if (sHi - sLo < EPS) continue;
    const ii = binCenters.flatMap((c, i) =>
      Math.min(sHi, c + step / 2) - Math.max(sLo, c - step / 2) > EPS ? [i] : []);
    if (ii.length) sectors.push({ lo: sLo, hi: sHi, ii });
  }
  const idx = sectors.map((s) => s.ii);
  return {
    centers: sectors.map((s) => (s.lo + s.hi) / 2),
    halfWidths: sectors.map((s) => (s.hi - s.lo) / 2),
    scores: idx.map((ii) => maxOf(ii.map((i) => record.scores[i] ?? 0))),
    minRange: idx.map((ii) => minNullable(ii.map((i) => record.min_range_m[i] ?? null))),
    confirmed: record.confirmed
      ? idx.map((ii) => ii.some((i) => record.confirmed![i]))
      : undefined,
    pObstacle: record.p_obstacle
      ? idx.map((ii) => maxOf(ii.map((i) => record.p_obstacle![i] ?? 0)))
      : undefined,
    threat: record.threat
      ? idx.map((ii) => maxOf(ii.map((i) => record.threat![i] ?? 0)))
      : undefined,
    sensorHits: idx.map((ii) => {
      const hit: [number, number, number] = [0, 0, 0];
      for (const i of ii) {
        const h = record.sensor_hits[i] ?? [0, 0, 0];
        hit[0] ||= h[0]!;
        hit[1] ||= h[1]!;
        hit[2] ||= h[2]!;
      }
      return hit;
    }),
    velocity: record.per_bin_velocity_mps
      ? idx.map((ii) => {
          const vals = ii
            .map((i) => record.per_bin_velocity_mps![i])
            .filter((v): v is number => v != null);
          return vals.length ? Math.max(...vals) : null; // + = approaching
        })
      : undefined,
    ttc: record.per_bin_ttc_s
      ? idx.map((ii) => minNullable(ii.map((i) => record.per_bin_ttc_s![i] ?? null)))
      : undefined,
  };
}

/** Sector edges in degrees for a width (drives the radar wedge grid). */
export function sectorEdges(record: SectorRecord, widthDeg: number): number[] {
  const v = sectorView(record, widthDeg);
  const edges = v.centers.map((c, i) => c - v.halfWidths[i]!);
  edges.push(v.centers[v.centers.length - 1]! + v.halfWidths[v.halfWidths.length - 1]!);
  return edges;
}

/** Pick the displayed per-sector series for a score source, falling back to
    the fused score when the clip has no scorer fields. */
export function valuesFor(
  v: SectorView,
  source: "scores" | "p_obstacle" | "threat",
): number[] {
  if (source === "p_obstacle" && v.pObstacle) return v.pObstacle;
  if (source === "threat" && v.threat) return v.threat;
  return v.scores;
}
