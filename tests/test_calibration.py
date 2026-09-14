import numpy as np
import pytest
import yaml

from scripts.utils.calibration import (
    get_dotted,
    load_detection,
    load_intrinsics,
    load_yaml,
    set_dotted,
)


def test_load_intrinsics_real_shapes(intrinsics_real):
    assert intrinsics_real["fisheye"]["K"].shape == (3, 3)
    assert intrinsics_real["fisheye"]["D"].shape in [(4, 1), (4,)]
    assert intrinsics_real["thermal"]["K"].shape == (3, 3)
    assert intrinsics_real["thermal"]["D"].size == 5


def test_load_detection_real_keys(detection_real):
    for k in ("horizon", "fisheye", "thermal", "mmwave", "fusion"):
        assert k in detection_real
    assert detection_real["fusion"]["bin_step_deg"] > 0


def test_load_yaml_passthrough(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("a: 1\nb: [2, 3]\n")
    out = load_yaml(p)
    assert out == {"a": 1, "b": [2, 3]}


def test_get_dotted_nested():
    cfg = {"a": {"b": {"c": 7}}}
    assert get_dotted(cfg, "a.b.c") == 7


def test_get_dotted_raises_on_missing():
    cfg = {"a": {}}
    with pytest.raises(KeyError):
        get_dotted(cfg, "a.b.c")


def test_set_dotted_modifies_in_place():
    cfg = {"a": {"b": {"c": 7}}}
    set_dotted(cfg, "a.b.c", 99)
    assert cfg["a"]["b"]["c"] == 99


def test_set_dotted_does_not_create_missing_keys():
    cfg = {"a": {}}
    with pytest.raises(KeyError):
        set_dotted(cfg, "a.b.c", 1)


def test_load_intrinsics_from_temp_yaml(tmp_path):
    cfg = {
        "fisheye": {
            "K": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]],
            "D": [[0.1], [0.0], [0.0], [0.0]],
            "image_size": [10, 10], "cx": 5, "pix_deg_ratio": 1.0, "model": "fisheye",
        },
        "thermal": {
            "K": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]],
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [10, 10], "cx": 5, "pix_deg_ratio": 1.0, "model": "pinhole",
        },
    }
    p = tmp_path / "intr.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = load_intrinsics(p)
    assert isinstance(out["fisheye"]["K"], np.ndarray)
    assert out["fisheye"]["K"].shape == (3, 3)


def test_load_intrinsics_skips_absent_sensor(tmp_path):
    """A yaml with only one sensor exercises the `sensor not in cfg` skip."""
    cfg = {
        "fisheye": {
            "K": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]],
            "D": [[0.1], [0.0], [0.0], [0.0]],
        },
        # No "thermal" key -> the loop's continue branch fires.
    }
    p = tmp_path / "intr_partial.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = load_intrinsics(p)
    assert isinstance(out["fisheye"]["K"], np.ndarray)
    assert "thermal" not in out
