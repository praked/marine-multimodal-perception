import numpy as np
import pytest
import yaml

from scripts.utils.geometry import (
    UP_LEVEL,
    _apply_transform,
    load_extrinsics,
    project_radar_to_fisheye,
    project_radar_to_thermal,
    range_from_undistorted_bbox,
    range_from_water_plane,
    range_from_water_plane_up,
    up_from_horizon_line,
    up_from_pitch_roll,
)


def _simple_P():
    """A plain pinhole matrix: fx=fy=500, principal point at (320, 240)."""
    return np.array([[500.0, 0.0, 320.0],
                     [0.0, 500.0, 240.0],
                     [0.0, 0.0, 1.0]])


def test_load_extrinsics_real(extrinsics_real):
    assert extrinsics_real["T_radar_to_fisheye"].shape == (4, 4)
    assert extrinsics_real["T_radar_to_thermal"].shape == (4, 4)
    assert "fisheye" in extrinsics_real["camera_height_m"]


def test_load_extrinsics_missing_file_returns_defaults(tmp_path):
    out = load_extrinsics(tmp_path / "nope.yaml")
    assert out["measured"] is False
    assert np.allclose(out["T_radar_to_fisheye"], np.eye(4))


def test_load_extrinsics_from_temp(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump({
        "measured": True,
        "T_radar_to_fisheye": np.eye(4).tolist(),
        "T_radar_to_thermal": np.eye(4).tolist(),
        "camera_height_m": {"fisheye": 0.5, "thermal": 0.5},
    }))
    out = load_extrinsics(p)
    assert out["measured"] is True


def test_load_extrinsics_missing_transform_keys_default_to_identity(tmp_path):
    """A present file lacking the T_* keys falls back to identity (line 53)."""
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump({"measured": True}))
    out = load_extrinsics(p)
    assert np.allclose(out["T_radar_to_fisheye"], np.eye(4))
    assert np.allclose(out["T_radar_to_thermal"], np.eye(4))
    # camera_height_m default also kicks in.
    assert out["camera_height_m"]["fisheye"] == pytest.approx(0.4)


def test_apply_transform_identity():
    pts = np.array([[1.0, 2.0, 3.0]])
    out = _apply_transform(pts, np.eye(4))
    np.testing.assert_allclose(out, pts)


def test_apply_transform_translation():
    pts = np.array([[1.0, 2.0, 3.0]])
    T = np.eye(4)
    T[:3, 3] = [10.0, 0.0, 0.0]
    out = _apply_transform(pts, T)
    np.testing.assert_allclose(out, [[11.0, 2.0, 3.0]])


def test_apply_transform_empty():
    out = _apply_transform(np.empty((0, 3)), np.eye(4))
    assert out.shape == (0, 3)


def test_project_radar_to_fisheye_empty(intrinsics_real, extrinsics_real):
    intr = intrinsics_real["fisheye"]
    T = extrinsics_real["T_radar_to_fisheye"]
    out = project_radar_to_fisheye(np.empty((0, 3)), intr["K"], intr["D"], T)
    assert out.shape == (0, 2)


def test_project_radar_to_fisheye_forward_lands_near_centre(intrinsics_real, extrinsics_real):
    """A point straight ahead (X=0, Y=5, Z=0) should project at fisheye image centre."""
    intr = intrinsics_real["fisheye"]
    T = extrinsics_real["T_radar_to_fisheye"]
    pts = np.array([[0.0, 5.0, 0.0]])
    out = project_radar_to_fisheye(pts, intr["K"], intr["D"], T)
    # Image center is at (cx_px, cy_px) approximately (444, 300).
    assert np.isfinite(out).all()
    assert abs(out[0, 0] - 444.11) < 5
    assert abs(out[0, 1] - 300.50) < 5


def test_project_radar_to_thermal_forward_lands_near_centre(intrinsics_real, extrinsics_real):
    intr = intrinsics_real["thermal"]
    T = extrinsics_real["T_radar_to_thermal"]
    pts = np.array([[0.0, 5.0, 0.0]])
    out = project_radar_to_thermal(pts, intr["K"], intr["D"], T)
    assert np.isfinite(out).all()
    assert abs(out[0, 0] - 80.44) < 5


def test_project_radar_behind_camera_returns_nan(intrinsics_real, extrinsics_real):
    """A point at Y < 0 (behind) should NOT project (NaN row)."""
    intr = intrinsics_real["fisheye"]
    T = extrinsics_real["T_radar_to_fisheye"]
    pts = np.array([[0.0, -1.0, 0.0]])
    out = project_radar_to_fisheye(pts, intr["K"], intr["D"], T)
    assert np.isnan(out).all()


def test_range_from_water_plane_basic(intrinsics_real):
    intr = intrinsics_real["fisheye"]
    bbox = (430, 340, 460, 360)   # bottom-centre below image centre -> water
    r = range_from_water_plane(bbox, intr["K"], intr["D"], camera_height_m=0.4,
                               model="fisheye")
    assert r is not None
    assert 0.1 < r < 100


def test_range_from_water_plane_above_horizon_returns_none(intrinsics_real):
    intr = intrinsics_real["fisheye"]
    # bbox in top half of image -> ray points up
    bbox = (430, 50, 460, 60)
    r = range_from_water_plane(bbox, intr["K"], intr["D"], camera_height_m=0.4,
                               model="fisheye")
    assert r is None


def test_range_from_water_plane_thermal_works(intrinsics_real):
    intr = intrinsics_real["thermal"]
    bbox = (70, 80, 90, 100)
    r = range_from_water_plane(bbox, intr["K"], intr["D"], camera_height_m=0.4,
                               model="pinhole")
    assert r is not None and r > 0


def test_range_from_water_plane_normalised_bbox(intrinsics_real):
    intr = intrinsics_real["fisheye"]
    r = range_from_water_plane((0.50, 0.55, 0.55, 0.60),
                               intr["K"], intr["D"], camera_height_m=0.4,
                               model="fisheye", image_size=(864, 648))
    assert r is not None


def test_range_from_water_plane_with_pitch(intrinsics_real):
    """A non-zero pitch should change the range estimate."""
    intr = intrinsics_real["fisheye"]
    bbox = (430, 340, 460, 360)
    r0 = range_from_water_plane(bbox, intr["K"], intr["D"], 0.4)
    r_pitch = range_from_water_plane(bbox, intr["K"], intr["D"], 0.4, pitch_rad=0.1)
    assert r0 != pytest.approx(r_pitch)


# --- range_from_undistorted_bbox (the camera-detection path) ---------------

def test_range_from_undistorted_below_centre_is_positive():
    P = _simple_P()
    # bottom-centre below the principal point -> ray points down to water
    r = range_from_undistorted_bbox((310, 300, 330, 320), P, camera_height_m=0.4)
    assert r is not None and r > 0


def test_range_from_undistorted_above_horizon_is_none():
    P = _simple_P()
    # bbox bottom ABOVE the principal point -> ray points up, no intersection
    r = range_from_undistorted_bbox((310, 100, 330, 120), P, camera_height_m=0.4)
    assert r is None


def test_range_from_undistorted_lower_in_frame_is_closer():
    """Physical sanity: a detection nearer the image bottom is nearer the boat."""
    P = _simple_P()
    near = range_from_undistorted_bbox((310, 460, 330, 470), P, 0.4)  # low in frame
    far = range_from_undistorted_bbox((310, 250, 330, 260), P, 0.4)   # near horizon
    assert near is not None and far is not None
    assert near < far


def test_range_from_undistorted_scales_with_height():
    """Doubling camera height doubles range for the same pixel (similar triangles)."""
    P = _simple_P()
    bbox = (310, 300, 330, 320)
    r1 = range_from_undistorted_bbox(bbox, P, camera_height_m=0.4)
    r2 = range_from_undistorted_bbox(bbox, P, camera_height_m=0.8)
    assert r2 == pytest.approx(2.0 * r1, rel=1e-9)


def test_range_from_undistorted_matches_pinhole_undistort():
    """With zero distortion, the undistorted path must equal range_from_water_plane
    fed the same matrix and D=0 (both are a plain pinhole back-projection)."""
    P = _simple_P()
    bbox = (310, 300, 330, 320)
    r_undist = range_from_undistorted_bbox(bbox, P, camera_height_m=0.4)
    r_ref = range_from_water_plane(bbox, P, np.zeros(5), camera_height_m=0.4,
                                   model="pinhole")
    assert r_undist == pytest.approx(r_ref, rel=1e-6)


def test_range_from_undistorted_degenerate_matrix_is_none():
    P = _simple_P()
    P[0, 0] = 0.0  # fx = 0
    assert range_from_undistorted_bbox((310, 300, 330, 320), P, 0.4) is None


def test_range_from_undistorted_negative_height_is_none():
    """A below-horizon ray (forward intersection) but a non-positive camera
    height drives t <= 0 -> None (geometry line 194)."""
    P = _simple_P()
    # bottom-centre below principal point -> ray[1] > 0 (passes horizon gate),
    # but height <= 0 makes t <= 0.
    r = range_from_undistorted_bbox((310, 300, 330, 320), P, camera_height_m=-0.4)
    assert r is None


# --- attitude-aware range (up vectors, horizon reference, clamp) ------------

def test_up_from_pitch_roll_level_is_up_level():
    np.testing.assert_allclose(up_from_pitch_roll(0.0, 0.0), UP_LEVEL, atol=1e-12)


def test_up_from_horizon_line_level_is_up_level():
    """A level horizon (slope 0, intercept = cy) must yield level-up."""
    P = _simple_P()
    cy = P[1, 2]
    up = up_from_horizon_line(0.0, cy, P)
    np.testing.assert_allclose(up, UP_LEVEL, atol=1e-9)


def test_range_up_level_matches_undistorted():
    """With level up, the up-based range equals range_from_undistorted_bbox."""
    P = _simple_P()
    bbox = (310, 300, 330, 320)
    r_up = range_from_water_plane_up(bbox, P, UP_LEVEL, camera_height_m=0.4)
    r_ref = range_from_undistorted_bbox(bbox, P, camera_height_m=0.4)
    assert r_up == pytest.approx(r_ref, rel=1e-9)


def test_range_up_matches_pitch_roll_rotation():
    """up_from_pitch_roll must reproduce the rotation-based range path."""
    P = _simple_P()
    bbox = (310, 320, 330, 340)
    up = up_from_pitch_roll(0.08, -0.05)
    r_up = range_from_water_plane_up(bbox, P, up, camera_height_m=0.4)
    r_rot = range_from_undistorted_bbox(bbox, P, camera_height_m=0.4,
                                        pitch_rad=0.08, roll_rad=-0.05)
    assert r_up is not None and r_rot is not None
    assert r_up == pytest.approx(r_rot, rel=1e-6)


def test_range_up_horizon_reference_changes_with_pitch():
    """Raising the detected horizon (boat pitched) shortens the range for a
    fixed detection pixel vs assuming the camera is level."""
    P = _simple_P()
    cy = P[1, 2]
    bbox = (310, 300, 330, 320)
    level = range_from_water_plane_up(bbox, P, up_from_horizon_line(0.0, cy, P), 0.4)
    # Horizon detected 40 px above centre -> the object is "more depressed".
    pitched = range_from_water_plane_up(bbox, P, up_from_horizon_line(0.0, cy - 40, P), 0.4)
    assert level is not None and pitched is not None
    assert pitched < level


def test_range_up_max_range_clamp_returns_none():
    P = _simple_P()
    cy = P[1, 2]
    # Just below the principal point -> a very long (near-horizon) range.
    bbox = (310, int(cy) + 2, 330, int(cy) + 3)
    assert range_from_water_plane_up(bbox, P, UP_LEVEL, 0.27, max_range_m=None) is not None
    assert range_from_water_plane_up(bbox, P, UP_LEVEL, 0.27, max_range_m=30.0) is None


def test_range_up_above_horizon_is_none():
    P = _simple_P()
    bbox = (310, 100, 330, 120)  # above centre with level up
    assert range_from_water_plane_up(bbox, P, UP_LEVEL, 0.4) is None


def test_up_from_horizon_line_degenerate_P_returns_up_level():
    """A zero P maps the horizon line to the zero vector (norm 0), so the
    function falls back to UP_LEVEL (line 281)."""
    P = np.zeros((3, 3))
    up = up_from_horizon_line(0.0, 0.0, P)
    np.testing.assert_allclose(up, UP_LEVEL)


def test_up_from_horizon_line_flips_when_pointing_down():
    """A camera matrix with negative fy makes the raw up vector point +Y, so
    the sign-correction branch flips it back to -Y (line 284)."""
    P = np.array([[500.0, 0.0, 320.0],
                  [0.0, -500.0, 240.0],
                  [0.0, 0.0, 1.0]])
    up = up_from_horizon_line(0.0, 0.0, P)
    # After the flip the vertical component must point "up" (-Y).
    assert up[1] < 0


def test_range_up_degenerate_matrix_is_none():
    """fx == 0 short-circuits range_from_water_plane_up (line 311)."""
    P = _simple_P()
    P[0, 0] = 0.0
    assert range_from_water_plane_up((310, 300, 330, 320), P, UP_LEVEL, 0.4) is None


def test_range_up_negative_height_is_none():
    """A below-horizon ray (denom < 0) with a non-positive height drives
    t <= 0 -> None (line 319)."""
    P = _simple_P()
    bbox = (310, 300, 330, 320)  # below centre -> denom < 0 with level up
    assert range_from_water_plane_up(bbox, P, UP_LEVEL, camera_height_m=-0.4) is None
