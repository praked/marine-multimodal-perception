/* Polar drawing geometry for the radar bird's-eye and the sector scorer.
   Data frame: X lateral (starboard +), Y forward, metres; bearing
   θ = atan2(X, Y) so 0° points up (dead ahead). Pure functions -> testable.

   The geometry mirrors poster/tools/render_hero.py so the web panels are
   1:1 with images/pipeline_system_in_action.png. */

export interface View {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  width: number; // px
  height: number; // px
}

export const RADAR_VIEW: View = {
  xMin: -7.5,
  xMax: 7.5,
  yMin: 0,
  yMax: 9.4,
  width: 600,
  height: 376,
};

export const SCORER_VIEW: View = {
  xMin: -9.0,
  xMax: 9.0,
  yMin: -0.2,
  yMax: 10.3,
  width: 720,
  height: 420,
};

export function toPx(x: number, y: number, view: View): [number, number] {
  const px = ((x - view.xMin) / (view.xMax - view.xMin)) * view.width;
  const py = view.height - ((y - view.yMin) / (view.yMax - view.yMin)) * view.height;
  return [px, py];
}

/** Point at bearing θ (deg, 0 = up, + = starboard) and radius r. */
export function polarPoint(bearingDeg: number, r: number): [number, number] {
  const rad = (bearingDeg * Math.PI) / 180;
  return [r * Math.sin(rad), r * Math.cos(rad)];
}

/** SVG polyline path along an arc of radius r between two bearings. */
export function arcPath(
  r: number,
  fromDeg: number,
  toDeg: number,
  view: View,
  stepDeg = 2,
): string {
  const parts: string[] = [];
  const n = Math.max(2, Math.ceil(Math.abs(toDeg - fromDeg) / stepDeg));
  for (let i = 0; i <= n; i++) {
    const theta = fromDeg + ((toDeg - fromDeg) * i) / n;
    const [x, y] = polarPoint(theta, r);
    const [px, py] = toPx(x, y, view);
    parts.push(`${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`);
  }
  return parts.join(" ");
}

/** Radial spoke from the origin to radius r at one bearing. */
export function spokePath(bearingDeg: number, r: number, view: View): string {
  const [ox, oy] = toPx(0, 0, view);
  const [x, y] = polarPoint(bearingDeg, r);
  const [px, py] = toPx(x, y, view);
  return `M${ox.toFixed(2)},${oy.toFixed(2)} L${px.toFixed(2)},${py.toFixed(2)}`;
}

/** Filled sector wedge: origin -> arc between the two bearings at radius r.
    `gapDeg` shaves each side (poster GAP = 0.9°) so sectors read as
    separate marks (the dataviz 2px-spacer rule, in angle space). */
export function sectorPath(
  centerDeg: number,
  halfWidthDeg: number,
  r: number,
  view: View,
  gapDeg = 0.9,
  stepDeg = 2,
): string {
  const from = centerDeg - halfWidthDeg + gapDeg;
  const to = centerDeg + halfWidthDeg - gapDeg;
  const [ox, oy] = toPx(0, 0, view);
  const parts = [`M${ox.toFixed(2)},${oy.toFixed(2)}`];
  const n = Math.max(2, Math.ceil(Math.abs(to - from) / stepDeg));
  for (let i = 0; i <= n; i++) {
    const theta = from + ((to - from) * i) / n;
    const [x, y] = polarPoint(theta, r);
    const [px, py] = toPx(x, y, view);
    parts.push(`L${px.toFixed(2)},${py.toFixed(2)}`);
  }
  parts.push("Z");
  return parts.join(" ");
}
