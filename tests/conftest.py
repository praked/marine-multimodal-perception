"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _has_footage() -> bool:
    """True when at least one real triplet is present under data/.

    The example footage is large and gitignored, so it is absent in CI and
    on fresh clones. Tests marked ``needs_data`` are skipped when it is."""
    try:
        from scripts.utils.datasets import list_triplets
        return len(list_triplets()) > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``needs_data`` tests when the example footage is absent,
    so the suite stays green in CI (which has no data/)."""
    if _has_footage():
        return
    skip = pytest.mark.skip(reason="needs data/ footage (gitignored; absent here)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def intrinsics_real():
    from scripts.utils.calibration import load_intrinsics
    return load_intrinsics()


@pytest.fixture
def detection_real():
    from scripts.utils.calibration import load_detection
    return load_detection()


@pytest.fixture
def extrinsics_real():
    from scripts.utils.geometry import load_extrinsics
    return load_extrinsics()


@pytest.fixture
def small_fisheye_frame():
    """100x100 BGR frame with a bright (but sub-glint) region producing
    a strong horizontal gradient."""
    img = np.full((100, 100, 3), 60, dtype=np.uint8)
    img[40:60, 40:60] = 180   # below the 240..255 glint inpaint range
    return img


@pytest.fixture
def small_thermal_frame():
    """100x100 BGR frame approximating a thermal image with a hot spot."""
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    img[40:60, 40:60] = 200
    return img


@pytest.fixture
def sample_radar_points():
    return np.array([
        [0.0, 1.5, 0.0],
        [0.5, 2.0, 0.0],
        [-0.3, 3.0, 0.0],
        [1.0, 4.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def boats_triplet():
    """First Boats clip; needs data/ to be present."""
    from scripts.utils.datasets import resolve_triplet
    try:
        return resolve_triplet(REPO_ROOT / "data" / "Boats" / "2025-06-23_16-21-07")
    except FileNotFoundError:
        pytest.skip("data/Boats/2025-06-23_16-21-07 not available")


@pytest.fixture
def make_fusion_triplet():
    """Factory: synthetic triplet whose radar CSV carries V+SNR+NOISE (the
    current capture format): the fusion-model test corpus builder."""
    import cv2

    from scripts.utils.datasets import Triplet

    def _make(base_dir: Path, ts: str, n: int = 6) -> "Triplet":
        scene_dir = base_dir / "Synth"
        scene_dir.mkdir(exist_ok=True, parents=True)
        fish = scene_dir / f"fisheye_{ts}.mp4"
        therm = scene_dir / f"thermal_{ts}.mp4"
        mm = scene_dir / f"mmwave_{ts}.csv"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fw = cv2.VideoWriter(str(fish), fourcc, 3.0, (160, 120))
        tw = cv2.VideoWriter(str(therm), fourcc, 3.0, (160, 120))
        for _ in range(n):
            f = np.full((120, 160, 3), 70, dtype=np.uint8)
            f[40:80, 60:100] = 200
            fw.write(f)
            tw.write(f)
        fw.release()
        tw.release()
        hms = ts.split("_")[1].split("-")
        base_s = int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
        rows = ["Date,Time,X,Y,Z,V,SNR,NOISE"]
        for i in range(n):
            t = base_s + i
            tstr = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.0"
            rows.append(f"{ts[:10]},{tstr},0.1,1.5,0.0,0.2,14.0,8.0")
        mm.write_text("\n".join(rows) + "\n")
        return Triplet(scene="Synth", timestamp=ts, fisheye=fish,
                       thermal=therm, mmwave=mm)

    return _make


@pytest.fixture
def synthetic_feature_root(tmp_path, intrinsics_real, detection_real,
                           make_fusion_triplet):
    """Two synthetic clips exported for real (build_features) + a labels
    JSONL: centred boat bbox frames 1-3 (near -> relevant), person on
    frame 2. Returns (features_root, loaded_labels, clip_ids)."""
    import json as _json

    from scripts.eval.metrics import load_labels
    from scripts.fusion_model.build_features import (
        build_for_triplet,
        write_tables,
    )

    root = tmp_path / "features"
    clip_ids = []
    for ts in ("2099-07-01_12-00-00", "2099-07-02_12-00-00"):
        trip = make_fusion_triplet(tmp_path, ts)
        bins_df, frames_df, meta = build_for_triplet(
            trip, intrinsics_real, detection_real, use_imu=False)
        write_tables(bins_df, frames_df, meta, root, fmt="csv")
        clip_ids.append(meta["clip_id"])

    lab_path = tmp_path / "labels.jsonl"
    recs = []
    for ts in ("2099-07-01_12-00-00", "2099-07-02_12-00-00"):
        for idx in (1, 2, 3):
            bboxes = [{"cls": "boat", "xyxy": [0.45, 0.55, 0.55, 0.80],
                       "confidence": 0.9}]
            if idx == 2:
                bboxes.append({"cls": "person",
                               "xyxy": [0.48, 0.60, 0.52, 0.78]})
            recs.append({"frame_id": f"Synth/{ts}/{idx:06d}",
                         "scene": "Synth", "audited": True,
                         "source": "manual", "width": 864, "height": 648,
                         "fisheye_bboxes": bboxes,
                         "obstacle_bins_fisheye": [0]})
    lab_path.write_text("\n".join(_json.dumps(r) for r in recs) + "\n")
    return root, load_labels([lab_path]), clip_ids


@pytest.fixture
def synthetic_triplet(tmp_path):
    """Construct a fake triplet on disk with 5 frames.

    Returns the resolved Triplet pointing to tmp_path/<scene>/<ts>...
    """
    import cv2

    from scripts.utils.datasets import Triplet

    scene_dir = tmp_path / "Synth"
    scene_dir.mkdir()
    ts = "2099-01-01_00-00-00"
    fish_path = scene_dir / f"fisheye_{ts}.mp4"
    therm_path = scene_dir / f"thermal_{ts}.mp4"
    mm_path = scene_dir / f"mmwave_{ts}.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish_path), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(therm_path), fourcc, 3.0, (160, 120))
    for i in range(5):
        f = np.full((120, 160, 3), 60 + i * 5, dtype=np.uint8)
        f[40:80, 60:100] = 200
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()

    rows = ["Date,Time,X,Y,Z"]
    for i in range(5):
        ts_str = f"00:00:{i:02d}.0"
        rows.append(f"2099-01-01,{ts_str},0.1,1.5,0.0")
        rows.append(f"2099-01-01,{ts_str},0.5,2.0,0.0")
    mm_path.write_text("\n".join(rows) + "\n")

    return Triplet(scene="Synth", timestamp=ts,
                   fisheye=fish_path, thermal=therm_path, mmwave=mm_path)
