/* Sun + moon horizontal coordinates for the replay scene. The solar side
   mirrors dashboard/tools/enrich_bundle.py (NOAA geometry); the lunar side
   is a truncated Meeus series — ±1-2° accuracy, plenty for placing a model
   in a 3D sky. Pure + tested. */

export interface HorizontalPos {
  elevationDeg: number;
  azimuthDeg: number; // from North, clockwise
}

const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;
const norm360 = (d: number) => ((d % 360) + 360) % 360;

function julianDays(date: Date): number {
  return date.getTime() / 86400000 + 2440587.5 - 2451545.0;
}

function gmstHours(d: number): number {
  return (18.697374558 + 24.06570982441908 * d) % 24;
}

function toHorizontal(
  raDeg: number,
  decRad: number,
  d: number,
  latDeg: number,
  lonDeg: number,
): HorizontalPos {
  const lst = norm360(gmstHours(d) * 15 + lonDeg);
  const ha = rad(norm360(lst - raDeg + 540) - 180);
  const lat = rad(latDeg);
  const elev = Math.asin(
    Math.sin(lat) * Math.sin(decRad) +
      Math.cos(lat) * Math.cos(decRad) * Math.cos(ha),
  );
  const az = Math.atan2(
    -Math.sin(ha),
    Math.tan(decRad) * Math.cos(lat) - Math.sin(lat) * Math.cos(ha),
  );
  return { elevationDeg: deg(elev), azimuthDeg: norm360(deg(az)) };
}

export function sunPosition(date: Date, latDeg: number, lonDeg: number): HorizontalPos {
  const d = julianDays(date);
  const g = rad(norm360(357.529 + 0.98560028 * d));
  const q = norm360(280.459 + 0.98564736 * d);
  const L = rad(norm360(q + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)));
  const e = rad(23.439 - 0.00000036 * d);
  const ra = norm360(deg(Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L))));
  const dec = Math.asin(Math.sin(e) * Math.sin(L));
  return toHorizontal(ra, dec, d, latDeg, lonDeg);
}

/** Truncated Meeus lunar position (main periodic terms only). */
export function moonPosition(date: Date, latDeg: number, lonDeg: number): HorizontalPos {
  const d = julianDays(date);
  const T = d / 36525;
  const Lp = norm360(218.3164477 + 481267.88123421 * T); // mean longitude
  const D = rad(norm360(297.8501921 + 445267.1114034 * T)); // elongation
  const M = rad(norm360(357.5291092 + 35999.0502909 * T)); // sun anomaly
  const Mp = rad(norm360(134.9633964 + 477198.8675055 * T)); // moon anomaly
  const F = rad(norm360(93.272095 + 483202.0175233 * T)); // arg latitude

  const lon =
    Lp +
    6.288774 * Math.sin(Mp) +
    1.274027 * Math.sin(2 * D - Mp) +
    0.658314 * Math.sin(2 * D) +
    0.213618 * Math.sin(2 * Mp) -
    0.185116 * Math.sin(M) -
    0.114332 * Math.sin(2 * F);
  const lat =
    5.128122 * Math.sin(F) +
    0.280602 * Math.sin(Mp + F) +
    0.277693 * Math.sin(Mp - F) +
    0.173237 * Math.sin(2 * D - F);

  const lonR = rad(norm360(lon));
  const latR = rad(lat);
  const e = rad(23.439 - 0.0000004 * d);
  const ra = norm360(
    deg(
      Math.atan2(
        Math.sin(lonR) * Math.cos(e) - Math.tan(latR) * Math.sin(e),
        Math.cos(lonR),
      ),
    ),
  );
  const dec = Math.asin(
    Math.sin(latR) * Math.cos(e) + Math.cos(latR) * Math.sin(e) * Math.sin(lonR),
  );
  return toHorizontal(ra, dec, d, latDeg, lonDeg);
}

/** Direction unit-vector in the scene frame (X = east, Y = up, Z = south). */
export function horizontalToVector(p: HorizontalPos): [number, number, number] {
  const el = rad(p.elevationDeg);
  const az = rad(p.azimuthDeg);
  return [
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    -Math.cos(az) * Math.cos(el),
  ];
}
