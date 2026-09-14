import numpy as np
import pytest
import yaml

from scripts.utils.clip_overrides import (
    apply_overrides,
    get_for_clip,
    load_overrides,
)


def test_load_overrides_real(repo_root):
    out = load_overrides(repo_root / "configs" / "clip_overrides.yaml")
    assert "Boats/2025-07-07_17-11-00" in out
    assert out["Boats/2025-07-07_17-11-00"]["fisheye"]["rotation_deg"] == 90


def test_load_overrides_missing(tmp_path):
    assert load_overrides(tmp_path / "absent.yaml") == {}


def test_load_overrides_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"; p.write_text("")
    assert load_overrides(p) == {}


def test_load_overrides_no_clips_key(tmp_path):
    p = tmp_path / "x.yaml"; p.write_text(yaml.safe_dump({"other": 1}))
    assert load_overrides(p) == {}


def test_get_for_clip_present():
    overrides = {"X/y": {"fisheye": {"rotation_deg": 90}}}
    assert get_for_clip(overrides, "X/y") == {"fisheye": {"rotation_deg": 90}}


def test_get_for_clip_absent():
    assert get_for_clip({}, "anything") == {}


def test_apply_overrides_passthrough_when_empty():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = apply_overrides(img, {})
    assert out is img
    out2 = apply_overrides(img, None)
    assert out2 is img


def test_apply_overrides_none_frame():
    assert apply_overrides(None, {"rotation_deg": 90}) is None


def test_apply_overrides_resize():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = apply_overrides(img, {"target_image_size": [864, 648]})
    assert out.shape == (648, 864, 3)


def test_apply_overrides_resize_noop_if_match():
    img = np.zeros((648, 864, 3), dtype=np.uint8)
    out = apply_overrides(img, {"target_image_size": [864, 648]})
    assert out.shape == (648, 864, 3)


def test_apply_overrides_rotation_90():
    img = np.zeros((648, 864, 3), dtype=np.uint8)
    img[10, 20] = 255
    out = apply_overrides(img, {"rotation_deg": 90})
    # 90 CW: new dims (648, 864) -> (864, 648) ... actually rotation swaps
    # original (h, w) = (648, 864) -> rotated (w, h) = (864, 648)
    # wait, cv2.ROTATE_90_CLOCKWISE: an HxW image becomes WxH.
    assert out.shape == (864, 648, 3)


def test_apply_overrides_rotation_180():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[5, 10] = 255
    out = apply_overrides(img, {"rotation_deg": 180})
    assert out.shape == (40, 60, 3)
    # 180 sends (5, 10) -> (40-1-5, 60-1-10) = (34, 49)
    assert out[34, 49, 0] == 255


def test_apply_overrides_rotation_270():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    out = apply_overrides(img, {"rotation_deg": 270})
    assert out.shape == (60, 40, 3)


def test_apply_overrides_rotation_0_is_noop():
    img = np.full((10, 20, 3), 7, dtype=np.uint8)
    out = apply_overrides(img, {"rotation_deg": 0})
    np.testing.assert_array_equal(out, img)


def test_apply_overrides_invalid_rotation_is_ignored():
    img = np.full((10, 20, 3), 7, dtype=np.uint8)
    out = apply_overrides(img, {"rotation_deg": 45})
    np.testing.assert_array_equal(out, img)


def test_apply_overrides_resize_then_rotate():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = apply_overrides(img, {"target_image_size": [864, 648], "rotation_deg": 90})
    # Resize first -> (648, 864), then 90 CW -> (864, 648)
    assert out.shape == (864, 648, 3)


def test_iterate_triplet_applies_overrides(intrinsics_real, detection_real,
                                            synthetic_triplet):
    """Pass an explicit override dict and confirm the yielded frames carry it."""
    from scripts.sensor_processing.pipeline import iterate_triplet
    overrides = {"fisheye": {"rotation_deg": 90}}
    for ts, fish, therm, mm in iterate_triplet(synthetic_triplet, detection_real,
                                               clip_overrides=overrides):
        # Synthetic frames are 120x160 (HxW). 90 CW -> 160x120.
        assert fish.shape[:2] == (160, 120)
        # Thermal had no override so it stays the same.
        assert therm.shape[:2] == (120, 160)
        break
