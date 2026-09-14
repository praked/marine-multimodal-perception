import { SEG_PALETTE } from "@/lib/palette";

/* Pure pixel routines for the eWaSR segmentation overlay. Inputs are plain
   arrays so everything here is unit-testable without a DOM canvas. */

export const OBSTACLE = 0;
export const WATER = 1;
export const SKY = 2;

/** Decode a mask image's RGBA pixels to class ids.
    Accepts label-encoded grayscale (0/1/2 in every channel) or the eWaSR
    colour palette; anything else maps to the nearest palette entry —
    mirroring scripts/utils/segmentation.decode_seg_array. */
export function decodeSeg(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): Uint8Array {
  const out = new Uint8Array(width * height);
  const palette = Object.entries(SEG_PALETTE).map(([k, v]) => ({
    cls: Number(k),
    rgb: v,
  }));
  for (let i = 0; i < width * height; i++) {
    const r = rgba[i * 4]!;
    const g = rgba[i * 4 + 1]!;
    const b = rgba[i * 4 + 2]!;
    if (r <= 2 && g <= 2 && b <= 2) {
      out[i] = r; // label-encoded grayscale
      continue;
    }
    let best = OBSTACLE;
    let bestDist = Infinity;
    for (const { cls, rgb } of palette) {
      const d =
        (r - rgb[0]) * (r - rgb[0]) +
        (g - rgb[1]) * (g - rgb[1]) +
        (b - rgb[2]) * (b - rgb[2]);
      if (d < bestDist) {
        bestDist = d;
        best = cls;
      }
    }
    out[i] = best;
  }
  return out;
}

/** RGBA overlay tinting each class with the eWaSR palette at `alpha`.
    Matches the Tkinter dashboard's 0.6*img + 0.4*colour blend when
    alpha = 0.4 (the canvas compositor does the blending). */
export function tintOverlay(
  cls: Uint8Array,
  width: number,
  height: number,
  alpha = 0.4,
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(new ArrayBuffer(width * height * 4));
  const a = Math.round(alpha * 255);
  for (let i = 0; i < width * height; i++) {
    const rgb = SEG_PALETTE[cls[i]!] ?? SEG_PALETTE[OBSTACLE]!;
    out[i * 4] = rgb[0];
    out[i * 4 + 1] = rgb[1];
    out[i * 4 + 2] = rgb[2];
    out[i * 4 + 3] = a;
  }
  return out;
}

/** Per-column topmost WATER row (the water edge), -1 where the column has no
    water. Mirrors poster/tools/render_hero.py water_edge_curve. */
export function waterEdgeColumns(
  cls: Uint8Array,
  width: number,
  height: number,
): Int16Array {
  const out = new Int16Array(width).fill(-1);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      if (cls[y * width + x] === WATER) {
        out[x] = y;
        break;
      }
    }
  }
  return out;
}

/** Median-smooth the edge with a horizontal window (poster: 9 px). */
export function smoothEdge(edge: Int16Array, window = 9): Int16Array {
  const half = Math.floor(window / 2);
  const out = new Int16Array(edge.length).fill(-1);
  const buf: number[] = [];
  for (let x = 0; x < edge.length; x++) {
    buf.length = 0;
    for (let k = x - half; k <= x + half; k++) {
      const v = edge[Math.min(Math.max(k, 0), edge.length - 1)]!;
      if (v >= 0) buf.push(v);
    }
    if (buf.length === 0) continue;
    buf.sort((a, b) => a - b);
    out[x] = buf[Math.floor(buf.length / 2)]!;
  }
  return out;
}

/** Split the edge into drawable polyline segments, skipping columns without
    water and breaking on vertical jumps (poster: |Δy| >= 40 at native res —
    scale-aware via `maxJump`). Returns arrays of [x, y] points. */
export function edgeSegments(
  edge: Int16Array,
  maxJump = 40,
): [number, number][][] {
  const segments: [number, number][][] = [];
  let current: [number, number][] = [];
  let prevY: number | null = null;
  for (let x = 0; x < edge.length; x++) {
    const y = edge[x]!;
    if (y < 0 || (prevY !== null && Math.abs(y - prevY) >= maxJump)) {
      if (current.length > 1) segments.push(current);
      current = [];
      prevY = y < 0 ? null : y;
      if (y >= 0) current.push([x, y]);
      continue;
    }
    current.push([x, y]);
    prevY = y;
  }
  if (current.length > 1) segments.push(current);
  return segments;
}
