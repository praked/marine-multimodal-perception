import type { Box } from "@/lib/types";

/* Pure geometry for the annotation stage. Boxes are normalised xyxy in the
   undistorted fisheye frame ([0,1], top-left -> bottom-right), matching
   labels/*.jsonl. All hit-testing happens in normalised space with pixel
   tolerances converted by the caller. */

export type Corner = "tl" | "tr" | "bl" | "br";

export interface HitResult {
  index: number;
  corner: Corner | null; // null = body hit (move)
}

export function clamp01(v: number): number {
  return Math.min(Math.max(v, 0), 1);
}

/** Normalise so x0<x1, y0<y1. */
export function normaliseBox(
  xyxy: [number, number, number, number],
): [number, number, number, number] {
  const [x0, y0, x1, y1] = xyxy;
  return [
    Math.min(x0, x1),
    Math.min(y0, y1),
    Math.max(x0, x1),
    Math.max(y0, y1),
  ];
}

export function cornerPoints(
  box: [number, number, number, number],
): Record<Corner, [number, number]> {
  const [x0, y0, x1, y1] = box;
  return { tl: [x0, y0], tr: [x1, y0], bl: [x0, y1], br: [x1, y1] };
}

/** Topmost (last-drawn-first) box whose corner or body contains (x, y).
    Corners take priority so small boxes stay resizable. `tol` is the corner
    pick radius in normalised units. */
export function hitTest(
  boxes: Box[],
  x: number,
  y: number,
  tol: number,
): HitResult | null {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const box = normaliseBox(boxes[i]!.xyxy);
    for (const [corner, [cx, cy]] of Object.entries(cornerPoints(box)) as [
      Corner,
      [number, number],
    ][]) {
      if (Math.abs(x - cx) <= tol && Math.abs(y - cy) <= tol) {
        return { index: i, corner };
      }
    }
  }
  for (let i = boxes.length - 1; i >= 0; i--) {
    const [x0, y0, x1, y1] = normaliseBox(boxes[i]!.xyxy);
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
      return { index: i, corner: null };
    }
  }
  return null;
}

/** Move a box by (dx, dy), clamped into the frame. */
export function moveBox(
  xyxy: [number, number, number, number],
  dx: number,
  dy: number,
): [number, number, number, number] {
  const [x0, y0, x1, y1] = xyxy;
  const w = x1 - x0;
  const h = y1 - y0;
  const nx0 = clamp01(Math.min(x0 + dx, 1 - w));
  const ny0 = clamp01(Math.min(y0 + dy, 1 - h));
  return [nx0, ny0, nx0 + w, ny0 + h];
}

/** Drag one corner to (x, y). */
export function resizeBox(
  xyxy: [number, number, number, number],
  corner: Corner,
  x: number,
  y: number,
): [number, number, number, number] {
  const [x0, y0, x1, y1] = xyxy;
  const cx = clamp01(x);
  const cy = clamp01(y);
  switch (corner) {
    case "tl":
      return normaliseBox([cx, cy, x1, y1]);
    case "tr":
      return normaliseBox([x0, cy, cx, y1]);
    case "bl":
      return normaliseBox([cx, y0, x1, cy]);
    case "br":
      return normaliseBox([x0, y0, cx, cy]);
  }
}

/** Reject degenerate boxes (mirrors the label tool's >=5 px rule; the
    threshold is normalised — caller passes 5/width, 5/height). */
export function isValidBox(
  xyxy: [number, number, number, number],
  minW: number,
  minH: number,
): boolean {
  const [x0, y0, x1, y1] = normaliseBox(xyxy);
  const eps = 1e-9;
  return x1 - x0 >= minW - eps && y1 - y0 >= minH - eps;
}

/** hitTest with the SELECTED box given priority: its corners, then its body,
    win over any overlapping box — resizing what you selected must not turn
    into selecting/moving a neighbour. Falls back to the normal topmost scan. */
export function hitTestPrioritised(
  boxes: Box[],
  selected: number | null,
  x: number,
  y: number,
  tol: number,
): HitResult | null {
  if (selected != null && boxes[selected]) {
    const own = hitTest([boxes[selected]!], x, y, tol);
    if (own) return { index: selected, corner: own.corner };
  }
  return hitTest(boxes, x, y, tol);
}

/** Indices of boxes whose CENTRE lies inside the (normalised) marquee rect.
    Centre containment feels right for "grab this cluster": a huge box that
    merely overlaps the corner of the marquee is not grabbed. */
export function boxesInMarquee(
  boxes: Box[],
  rect: [number, number, number, number],
): number[] {
  const [x0, y0, x1, y1] = normaliseBox(rect);
  const out: number[] = [];
  boxes.forEach((b, i) => {
    const [bx0, by0, bx1, by1] = normaliseBox(b.xyxy);
    const cx = (bx0 + bx1) / 2;
    const cy = (by0 + by1) / 2;
    if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) out.push(i);
  });
  return out;
}
