"""Tests for scripts/eval/association_report.py: pixel-association A/B report.

run()/main() are driven over the synthetic triplet with the pipeline swapped
for a deterministic stub whose detections sit exactly on the real projected
radar pixels (real intrinsics + measured extrinsics), so the report's own
re-association genuinely matches. Pipeline detector behaviour is covered by
test_pipeline; association mechanics by test_association."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.eval.association_report as ar
from scripts.utils.calibration import load_intrinsics
from scripts.utils.cv_common import pinhole_new_K
from scripts.utils.geometry import load_extrinsics, project_radar_to_undistorted

# The conftest synthetic_triplet's radar rows (per frame).
PTS = np.array([[0.1, 1.5, 0.0], [0.5, 2.0, 0.0]])


def test_stats_empty():
    assert ar._stats([]) == "n=0"


def test_stats_values():
    s = ar._stats([1.0, 2.0, 3.0])
    assert "n=3" in s and "median +2.00" in s and "mean +2.00" in s


class _StubPipeline:
    """Two matched fisheye detections on the projected pixel of the nearest
    radar return (one without a monocular range), one unmatched far away;
    one matched thermal detection. Frame 1 exercises the skip branches
    (empty radar_ranges / thermal result absent)."""

    def __init__(self, intrinsics, detection, attitude_provider=None):
        self.extrinsics = load_extrinsics()
        self._n = 0
        px_f = project_radar_to_undistorted(
            PTS, np.asarray(intrinsics["fisheye"]["K"], float),
            self.extrinsics["T_radar_to_fisheye"])
        self._fish_xy = (int(px_f[0, 0]), int(px_f[0, 1]))
        P_t = pinhole_new_K(intrinsics["thermal"]["K"],
                            intrinsics["thermal"]["D"], (160, 120))
        px_t = project_radar_to_undistorted(
            PTS, P_t, self.extrinsics["T_radar_to_thermal"])
        self._therm_xy = (int(px_t[0, 0]), int(px_t[0, 1]))

    def process_frame(self, fish, therm, pts, timestamp=None, frame_id=None):
        i = self._n
        self._n += 1
        mm = SimpleNamespace(points_xyz=np.asarray(pts, float)[:, :3])
        if i == 1:
            fisheye = SimpleNamespace(
                coords=[(5, 5)], sizes=[10.0], angles=[30.0], ranges=[2.0],
                mono_ranges=[2.0], radar_ranges=[], undistorted=None)
            thermal = None
        else:
            fisheye = SimpleNamespace(
                coords=[self._fish_xy, self._fish_xy, (5, 5)],
                sizes=[20.0, 20.0, 20.0],
                angles=[3.0, 3.5, 30.0],
                ranges=[1.6, 1.5, 2.0],
                mono_ranges=[1.6, None, 2.0],
                radar_ranges=[1.5, 1.5, None],
                undistorted=None)
            thermal = SimpleNamespace(
                coords=[self._therm_xy], sizes=[10.0], angles=[1.0],
                ranges=[1.4], mono_ranges=None, radar_ranges=[1.3],
                undistorted=np.zeros((120, 160, 3), np.uint8))
        return SimpleNamespace(
            fisheye=fisheye, thermal=thermal, mmwave=mm,
            fusion=SimpleNamespace(confirmed=[True, False]), timestamp=timestamp)


class _EmptyPipeline:
    """No detections at all: covers the n_det == 0 / no-stats branches."""

    def __init__(self, intrinsics, detection, attitude_provider=None):
        self.extrinsics = load_extrinsics()
        self.attitude = attitude_provider

    def process_frame(self, fish, therm, pts, timestamp=None, frame_id=None):
        return SimpleNamespace(
            fisheye=SimpleNamespace(coords=[], sizes=[], angles=[], ranges=[],
                                    mono_ranges=[], radar_ranges=[],
                                    undistorted=None),
            thermal=None,
            mmwave=SimpleNamespace(points_xyz=np.asarray(pts, float)[:, :3]),
            fusion=SimpleNamespace(confirmed=[]), timestamp=timestamp)


def _quad_prefix(tmp_path: Path, n: int = 4) -> Path:
    """Synthetic quad triplet (with a native-Euler IMU sidecar) for --imu."""
    import cv2

    ts = "2099-07-08_12-00-00"
    scene = tmp_path / "2099-07-08"
    scene.mkdir()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(scene / f"fisheye_{ts}.mp4"), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(scene / f"thermal_{ts}.mp4"), fourcc, 3.0, (160, 120))
    for _ in range(n):
        f = np.full((120, 160, 3), 70, dtype=np.uint8)
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()
    mm = ["Date,Time,X,Y,Z"]
    imu = ["Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az"]
    for i in range(n):
        t = f"12:00:{i:02d}.0"
        mm.append(f"2099-07-08,{t},0.1,1.5,0.0")
        imu.append(f"2099-07-08,{t},0.0,1.0,2.0,0.0,0.0,9.8")
    (scene / f"mmwave_{ts}.csv").write_text("\n".join(mm) + "\n")
    (scene / f"imu_{ts}.csv").write_text("\n".join(imu) + "\n")
    return scene / ts


class TestRun:
    def test_matches_and_confirmed_bins(self, synthetic_triplet):
        from scripts.utils.calibration import load_detection
        intrinsics = load_intrinsics()
        detection = load_detection()
        detection["fusion"].setdefault("association", {})["enabled"] = True
        # run() constructs the REAL pipeline here (detector behaviour itself
        # is test_pipeline's job): this checks the accounting is consistent
        # on whatever the real pipeline produced.
        out = ar.run(synthetic_triplet, detection, intrinsics, None,
                     max_frames=0)
        assert out["n_frames"] == 5
        for cam in ("fisheye", "thermal"):
            assert out[cam]["n_matched"] <= out[cam]["n_det"]
        assert out["confirmed_bins_per_frame"] >= 0.0


class TestMain:
    def test_main_reports_both_models(self, monkeypatch, synthetic_triplet,
                                      capsys):
        monkeypatch.setattr(ar, "ObstacleDetectionPipeline", _StubPipeline)
        prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
        rc = ar.main(["--triplet", str(prefix)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "bearing_model = linear" in out
        assert "bearing_model = pinhole" in out
        assert "(5 frames" in out
        # fisheye: 3 detections/frame x 4 full frames + 1 on frame 1 = 13;
        # 2 matched per full frame = 8
        assert "fisheye: 13 detections, 8 radar-matched (61.5%)" in out
        # thermal detection sits on its projected radar pixel -> matched
        assert "thermal: 4 detections, 4 radar-matched (100.0%)" in out
        assert "bearing residual (cam - radar, deg)" in out
        assert "mono vs radar range" in out
        assert "<=20%" in out

    def test_main_imu_seg_max_frames(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(ar, "ObstacleDetectionPipeline", _EmptyPipeline)
        prefix = _quad_prefix(tmp_path)
        rc = ar.main(["--triplet", str(prefix), "--imu", "--seg",
                      "--max-frames", "2"])
        assert rc == 0
        out = capsys.readouterr().out
        # max_frames stops after 2 processed frames (counter reads 3)
        assert "(3 frames" in out
        # no detections: rate falls back to 0.0 and stats print n=0
        assert "fisheye: 0 detections, 0 radar-matched (0.0%)" in out
        assert "n=0" in out
        assert "mono vs radar range" not in out
