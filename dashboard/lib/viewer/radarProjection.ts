/* Radar → undistorted-fisheye projection, the TS twin of
   scripts/utils/geometry.py project_radar_to_undistorted with the
   pipeline's mount-yaw correction in front (pipeline.py process_mmwave).

   The bundle's radar.json holds RAW capture points (TI frame: X lateral
   right+, Y forward, Z up, metres) — the mount-yaw rotation is applied at
   processing time, so it must be mirrored here. The bundle's fisheye
   frames are the UNDISTORTED image scaled to 432×324, and undistortion
   remaps with K as the new camera matrix, so projection is pure pinhole
   with K and pixels normalise by the native 864×648.

   Constants are copies with provenance (re-copy if these files change):
   - K: configs/intrinsics.yaml fisheye.K (V.2-confirmed 2026-08-17)
   - T: configs/extrinsics.yaml T_radar_to_fisheye (measured 2026-07-07,
        day-1 orientation restored 2026-08-14)
   - mount yaw: configs/detection.yaml mmwave.mount_yaw_deg (bearing walk,
        −6.0° ± 0.8°, 2026-08-19) */

const K = { fx: 418.51, fy: 418.58, cx: 444.11, cy: 300.5 };
/** Row-major 3×4 of configs/extrinsics.yaml T_radar_to_fisheye. */
const T = [
  [1.0, 0.0, 0.0, -0.0516],
  [0.0, 0.0, -1.0, 0.002],
  [0.0, 1.0, 0.0, -0.0043],
] as const;
export const MOUNT_YAW_DEG = -6.0;
const NATIVE_W = 864;
const NATIVE_H = 648;

export interface ProjectedReturn {
  /** Normalised image coords (0..1 of the undistorted frame). */
  x: number;
  y: number;
  /** Forward range (m) after mount-yaw correction — matches the pipeline's
      y_min/y_max filters. */
  rangeY: number;
  /** Radial Doppler (m/s, raw sign: approach negative), NaN when absent. */
  doppler: number;
}

/** Project one frame's radar points ([x,y,z] or [x,y,z,v] rows) into the
    undistorted fisheye. Points behind the camera or outside the frame are
    dropped; `yMin` mirrors the display clutter filter. */
export function projectRadarToFisheye(
  points: number[][],
  yMin = 0,
): ProjectedReturn[] {
  const th = (-MOUNT_YAW_DEG * Math.PI) / 180; // correct = rotate by −yaw
  const c = Math.cos(th);
  const s = Math.sin(th);
  const out: ProjectedReturn[] = [];
  for (const p of points) {
    if (p.length < 3) continue;
    const xr = p[0]! * c + p[1]! * s;
    const yr = -p[0]! * s + p[1]! * c;
    const zr = p[2]!;
    if (yr < yMin) continue;
    const camX = T[0][0] * xr + T[0][1] * yr + T[0][2] * zr + T[0][3];
    const camY = T[1][0] * xr + T[1][1] * yr + T[1][2] * zr + T[1][3];
    const camZ = T[2][0] * xr + T[2][1] * yr + T[2][2] * zr + T[2][3];
    if (camZ <= 1e-6) continue;
    const u = (K.fx * camX) / camZ + K.cx;
    const v = (K.fy * camY) / camZ + K.cy;
    if (u < 0 || u >= NATIVE_W || v < 0 || v >= NATIVE_H) continue;
    out.push({
      x: u / NATIVE_W,
      y: v / NATIVE_H,
      rangeY: yr,
      doppler: p.length > 3 && p[3] != null ? p[3]! : NaN,
    });
  }
  return out;
}
