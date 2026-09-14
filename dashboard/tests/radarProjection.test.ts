import { describe, expect, it } from "vitest";
import { projectRadarToFisheye } from "@/lib/viewer/radarProjection";

/* Fixture values generated from the repo's Python geometry
   (scripts/utils/geometry.py project_radar_to_undistorted with the
   pipeline's mount-yaw −6.0° correction, K from configs/intrinsics.yaml,
   T_radar_to_fisheye from configs/extrinsics.yaml):

     [ 1.0, 5.0,  0.2] -> px (570.2163, 283.4600) -> (0.6600, 0.4374)
     [-2.0, 8.0, -0.1] -> px (382.3450, 305.7317) -> (0.4425, 0.4718)
     [ 0.0, 3.0,  0.0] -> px (480.9122, 300.7810) -> (0.5566, 0.4642)
     [ 0.5,-1.0,  0.0] -> behind the camera -> dropped                */

describe("projectRadarToFisheye", () => {
  it("matches the Python geometry fixture", () => {
    const out = projectRadarToFisheye([
      [1.0, 5.0, 0.2],
      [-2.0, 8.0, -0.1],
      [0.0, 3.0, 0.0],
      [0.5, -1.0, 0.0],
    ]);
    expect(out).toHaveLength(3); // behind-camera point dropped
    expect(out[0]!.x).toBeCloseTo(0.66, 3);
    expect(out[0]!.y).toBeCloseTo(0.4374, 3);
    expect(out[1]!.x).toBeCloseTo(0.4425, 3);
    expect(out[1]!.y).toBeCloseTo(0.4718, 3);
    expect(out[2]!.x).toBeCloseTo(0.5566, 3);
    expect(out[2]!.y).toBeCloseTo(0.4642, 3);
  });

  it("carries Doppler when present, NaN otherwise", () => {
    const [withV] = projectRadarToFisheye([[1.0, 5.0, 0.2, -0.4]]);
    expect(withV!.doppler).toBe(-0.4);
    const [without] = projectRadarToFisheye([[1.0, 5.0, 0.2]]);
    expect(Number.isNaN(without!.doppler)).toBe(true);
  });

  it("yMin mirrors the display clutter filter on the corrected range", () => {
    // yr after the −6° mount-yaw correction of [0, 0.4, 0] is ~0.398 m
    expect(projectRadarToFisheye([[0.0, 0.4, 0.0]], 0.5)).toHaveLength(0);
    expect(projectRadarToFisheye([[0.0, 0.4, 0.0]], 0)).toHaveLength(1);
  });

  it("drops rows outside the frame and malformed rows", () => {
    expect(projectRadarToFisheye([[9.0, 0.5, 0.0]])).toHaveLength(0); // far left/right
    expect(projectRadarToFisheye([[1.0]])).toHaveLength(0);
    expect(projectRadarToFisheye([])).toHaveLength(0);
  });
});
