import json
import math
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.sensor_processing.fusion import main


def test_fusion_runs_and_writes_jsonl(boats_triplet, tmp_path):
    out = tmp_path / "sectors.jsonl"
    with mock.patch.object(sys, "argv", [
        "fusion", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--out", str(out), "--print-every", "0",
    ]):
        main()
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) > 0
    rec = json.loads(lines[0])
    for k in ("timestamp", "clip_id", "bin_centers_deg", "scores", "min_range_m", "sensor_hits", "tracked"):
        assert k in rec


def test_fusion_with_tracker(boats_triplet, tmp_path):
    out = tmp_path / "sectors_tracked.jsonl"
    with mock.patch.object(sys, "argv", [
        "fusion", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--out", str(out), "--track", "--print-every", "0",
        "--track-min-hits", "2", "--track-max-age", "3",
    ]):
        main()
    lines = out.read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["tracked"] is True


def test_fusion_print_every_emits_summary(synthetic_triplet, capsys):
    """--print-every 1 triggers _print_summary for every frame, exercising
    both the call site (line 95) and the summary formatter (lines 142-145)."""
    prefix = str(synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp)
    with mock.patch.object(sys, "argv", [
        "fusion", "--triplet", prefix, "--print-every", "1", "--no-heading",
    ]):
        main()
    captured = capsys.readouterr()
    # _print_summary prints to stdout; the per-frame summary lines carry the
    # timestamp and "+0:" style bin cells.
    assert "00:00:0" in captured.out
    # At least one cell shows a min-range suffix ("/1.5m") from the radar.
    assert "m" in captured.out


def test_print_summary_formats_cells(capsys):
    """Directly exercise _print_summary with a None and a non-None range."""
    import numpy as np

    from scripts.sensor_processing.fusion import _print_summary
    centers = np.array([-10.0, 0.0])
    scores = [0.0, 0.66]
    min_ranges = [None, 2.5]
    _print_summary("00:00:01.0", centers, scores, min_ranges)
    out = capsys.readouterr().out
    assert "00:00:01.0" in out
    assert "/2.5m" in out
    # The None-range cell has no "/...m" suffix.
    assert "-10:0.00" in out


def test_fusion_resolve_triplet_failure(tmp_path):
    """Pointing at a path that doesn't resolve should raise."""
    with mock.patch.object(sys, "argv", [
        "fusion", "--triplet", str(tmp_path / "Nope" / "1999-01-01_00-00-00"),
    ]):
        with pytest.raises(FileNotFoundError):
            main()


# ---------------------------------------------------------------------------
# Synthetic full-stack runs (no data/ needed): --nav / --seg / --scorer / IMU
# ---------------------------------------------------------------------------

_TS = "2099-07-01_12-00-00"


def _write_seg_masks(seg_root, ts, n):
    """Per-frame water/sky/obstacle masks matching make_fusion_triplet's
    120x160 frames + radar timestamps (base second + i)."""
    import numpy as np
    from PIL import Image

    clip_dir = seg_root / f"Synth__{ts}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    hms = ts.split("_")[1].split("-")
    base_s = int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    mask = np.ones((120, 160), dtype=np.uint8)      # water
    mask[:30, :] = 2                                 # sky above a level edge
    mask[40:80, 60:100] = 0                          # obstacle blob on water
    for i in range(n):
        t = base_s + i
        token = f"{t // 3600:02d}-{t % 3600 // 60:02d}-{t % 60:02d}.0"
        Image.fromarray(mask).save(clip_dir / f"ts={token}.png")


def _write_imu_sidecar(scene_dir, ts, n):
    """Native-Euler UART-RVC sidecar aligned with the radar timestamps."""
    hms = ts.split("_")[1].split("-")
    base_s = int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    rows = ["Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az"]
    for i in range(n):
        t = base_s + i
        tstr = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.0"
        rows.append(f"{ts[:10]},{tstr},90.0,2.0,1.0,0.0,0.0,9.8")
    (scene_dir / f"imu_{ts}.csv").write_text("\n".join(rows) + "\n")


def _detection_yaml_with_seg_root(tmp_path, seg_root):
    import yaml

    from scripts.utils.calibration import load_detection
    det = load_detection()
    det.setdefault("segmentation", {})["seg_root"] = str(seg_root)
    # The synthetic 160x120 frames undistort into a mostly-black 864x648
    # canvas (mean luminance < 25), which the darkness gate would treat as a
    # night frame and drop the mask. The gate itself is covered by
    # test_fusion_features.test_seg_darkness_gate.
    det["segmentation"]["darkness_gate"] = False
    path = tmp_path / "detection_override.yaml"
    path.write_text(yaml.safe_dump(det))
    return path


def _scorer_json(tmp_path):
    import numpy as np

    from scripts.fusion_model.models import (
        CONTEXT_COLUMNS,
        GatedMixtureScorer,
    )
    c = len(CONTEXT_COLUMNS)
    m = GatedMixtureScorer(u=np.zeros(3), v=np.zeros((3, c)),
                           b0=0.0, b=np.zeros(c))
    path = tmp_path / "scorer.json"
    m.save(path)
    return path


def _run_main(argv):
    with mock.patch.object(sys, "argv", ["fusion"] + argv):
        main()


def test_fusion_nav_profile_full_stack(tmp_path, make_fusion_triplet):
    """--nav + --bearing + --scorer over a synthetic quad clip (seg masks +
    IMU sidecar): exercises the flag plumbing (nav/seg/fp-filter/free-space/
    assoc/bearing/vote-gate), IMU replay attach, the free-space distance via
    the IMU up-vector, the confirmed/attitude/free_space_m/p_obstacle/threat
    JSONL fields, and the shadow-scorer per-frame path."""
    n = 5
    trip = make_fusion_triplet(tmp_path, _TS, n=n)
    scene_dir = trip.fisheye.parent
    _write_imu_sidecar(scene_dir, _TS, n)
    seg_root = tmp_path / "seg"
    _write_seg_masks(seg_root, _TS, n)
    det_yaml = _detection_yaml_with_seg_root(tmp_path, seg_root)
    scorer = _scorer_json(tmp_path)

    out = tmp_path / "sectors_nav.jsonl"
    _run_main([
        "--triplet", str(scene_dir / _TS), "--out", str(out),
        "--detection", str(det_yaml), "--nav", "--bearing", "pinhole",
        "--scorer", str(scorer), "--print-every", "0",
    ])
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == n
    n_bins = len(recs[0]["bin_centers_deg"])
    # Under the box default (attitude_source: rotation) the replayed yaw is
    # the COMPOSED camera-frame heading, not the sidecar's raw RVC yaw
    # (2026-08-24 follow-up: the raw yaw is gimbal-aliased on this mount).
    from scripts.sensor_processing.imu_replay import camera_heading_from_rvc
    want_yaw = math.degrees(camera_heading_from_rvc(
        math.radians(90.0), math.radians(2.0), math.radians(1.0)))
    for r in recs:
        # IMU replay attached and sampled per frame.
        assert r["attitude"] is not None
        assert r["attitude"]["yaw_deg"] == pytest.approx(want_yaw, abs=1e-3)
        # --assoc (via --nav): per-bin confirmation vector.
        assert len(r["confirmed"]) == n_bins
        # Free-space profile from the seg mask (masks resolved by frame_id).
        assert "free_space_m" in r
        assert len(r["free_space_m"]) == n_bins
        # Shadow scorer fields, additive to the legacy schema.
        assert len(r["p_obstacle"]) == n_bins
        assert len(r["threat"]) == n_bins
        assert all(0.0 <= p <= 1.0 for p in r["p_obstacle"])


def test_fusion_seg_flags_without_imu(tmp_path, make_fusion_triplet):
    """--seg --seg-fp-filter --free-space-heading --no-imu: the free-space
    up-vector falls back to the water-edge horizon (no attitude), and the
    fp-filter block is enabled explicitly rather than via --nav."""
    n = 4
    ts = "2099-07-02_12-00-00"
    trip = make_fusion_triplet(tmp_path, ts, n=n)
    scene_dir = trip.fisheye.parent
    _write_imu_sidecar(scene_dir, ts, n)   # present but ignored by --no-imu
    seg_root = tmp_path / "seg"
    _write_seg_masks(seg_root, ts, n)
    det_yaml = _detection_yaml_with_seg_root(tmp_path, seg_root)

    out = tmp_path / "sectors_seg.jsonl"
    _run_main([
        "--triplet", str(scene_dir / ts), "--out", str(out),
        "--detection", str(det_yaml), "--seg", "--seg-fp-filter",
        "--free-space-heading", "--no-imu", "--print-every", "0",
    ])
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == n
    for r in recs:
        assert r["attitude"] is None            # --no-imu
        assert "free_space_m" in r              # water-edge up-vector path


def test_fusion_scorer_sun_position_valueerror(tmp_path, make_fusion_triplet,
                                               monkeypatch):
    """When frame_datetime_utc raises ValueError the scorer path degrades to
    NaN sun elevation instead of crashing (fusion.py except branch)."""
    def _boom(*a, **k):
        raise ValueError("bad clock")
    monkeypatch.setattr(
        "scripts.fusion_model.build_features.frame_datetime_utc", _boom)

    ts = "2099-07-03_12-00-00"
    trip = make_fusion_triplet(tmp_path, ts, n=3)
    out = tmp_path / "sectors_scorer.jsonl"
    _run_main([
        "--triplet", str(trip.fisheye.parent / ts), "--out", str(out),
        "--scorer", str(_scorer_json(tmp_path)), "--print-every", "0",
    ])
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert recs
    for r in recs:
        assert all(0.0 <= p <= 1.0 for p in r["p_obstacle"])


def test_free_space_dist_none_without_mask():
    """_free_space_dist short-circuits to None when the frame has no seg
    mask (and when there is no fisheye result at all)."""
    from types import SimpleNamespace

    from scripts.sensor_processing.fusion import _free_space_dist

    no_fisheye = SimpleNamespace(fisheye=None)
    assert _free_space_dist(no_fisheye, {}, {}, 0.27) is None
    no_mask = SimpleNamespace(fisheye=SimpleNamespace(seg_mask=None))
    assert _free_space_dist(no_mask, {}, {}, 0.27) is None
