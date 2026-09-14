"""Tests for scripts/eval/imu_attitude_check.py: IMU vs horizon mapping check.

The unit helpers (pitch_roll_from_horizon / _centred_corr) are tested
directly; main() runs against a synthetic quad triplet (fisheye + thermal +
radar + imu Euler sidecar) with detect_horizon / undistort_fisheye swapped
for deterministic stubs: real horizon behaviour is covered elsewhere
(test_pipeline); here we exercise the pairing gates + mapping scoring."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import scripts.eval.imu_attitude_check as iac
from scripts.utils.calibration import load_intrinsics
from scripts.utils.geometry import horizon_line_from_up, up_from_pitch_roll

K_TOY = np.array([[400.0, 0.0, 432.0],
                  [0.0, 400.0, 324.0],
                  [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestCentredCorr:
    def test_identical_traces_full_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # median == mean -> exactly 1
        assert iac._centred_corr(a, a) == pytest.approx(1.0)

    def test_negated_traces_anticorrelated(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert iac._centred_corr(a, -a) == pytest.approx(-1.0)

    def test_constant_trace_returns_zero(self):
        a = np.full(5, 3.0)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert iac._centred_corr(a, b) == 0.0
        assert iac._centred_corr(b, a) == 0.0


class TestPitchRollFromHorizon:
    def test_level_horizon_is_zero_attitude(self):
        # Level camera: horizon at row cy, slope 0.
        p, r = iac.pitch_roll_from_horizon(0.0, float(K_TOY[1, 2]), K_TOY)
        assert p == pytest.approx(0.0, abs=1e-9)
        assert r == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("pitch_deg,roll_deg",
                             [(2.0, 5.0), (-3.0, -4.0), (1.5, 0.0), (0.0, -6.0)])
    def test_round_trip_from_synthetic_attitude(self, pitch_deg, roll_deg):
        p_in, r_in = math.radians(pitch_deg), math.radians(roll_deg)
        slope, intercept = horizon_line_from_up(
            up_from_pitch_roll(p_in, r_in), K_TOY)
        p_out, r_out = iac.pitch_roll_from_horizon(slope, intercept, K_TOY)
        assert p_out == pytest.approx(p_in, abs=1e-6)
        assert r_out == pytest.approx(r_in, abs=1e-6)


# ---------------------------------------------------------------------------
# Synthetic quad triplet
# ---------------------------------------------------------------------------

TS = "2099-07-08_12-00-00"


def _make_quad(tmp_path: Path, n: int, rolls_rad=None, pitches_rad=None,
               imu_skip=frozenset()) -> Path:
    """Quad triplet on disk: n frames at 1 s spacing + native-Euler IMU
    sidecar. Returns the --triplet prefix. rolls/pitches index by frame."""
    import cv2

    scene = tmp_path / "2099-07-08"
    scene.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(scene / f"fisheye_{TS}.mp4"), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(scene / f"thermal_{TS}.mp4"), fourcc, 3.0, (160, 120))
    for _ in range(n):
        f = np.full((120, 160, 3), 70, dtype=np.uint8)
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()

    mm_rows = ["Date,Time,X,Y,Z"]
    imu_rows = ["Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az"]
    for i in range(n):
        t = f"12:00:{i:02d}.0"
        mm_rows.append(f"2099-07-08,{t},0.1,1.5,0.0")
        if i in imu_skip:
            continue
        r = math.degrees(rolls_rad[i]) if rolls_rad else 0.0
        p = math.degrees(pitches_rad[i]) if pitches_rad else 0.0
        imu_rows.append(f"2099-07-08,{t},0.0,{p:.6f},{r:.6f},0.0,0.0,9.8")
    (scene / f"mmwave_{TS}.csv").write_text("\n".join(mm_rows) + "\n")
    (scene / f"imu_{TS}.csv").write_text("\n".join(imu_rows) + "\n")
    return scene / TS


def _stub_horizon(monkeypatch, schedule):
    """Replace detect_horizon with a per-call schedule of
    (slope, intercept, confidence); undistort becomes identity."""
    calls = {"i": -1}

    def fake_detect(frame, **kwargs):
        calls["i"] += 1
        s, b, c = schedule[calls["i"]]
        return np.zeros((2, 2), np.uint8), s, b, c

    monkeypatch.setattr(iac, "detect_horizon", fake_detect)
    monkeypatch.setattr(iac, "undistort_fisheye", lambda f, K, D: f)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_end_to_end_recovers_identity_mapping(self, tmp_path, monkeypatch,
                                                  capsys):
        K = load_intrinsics()["fisheye"]["K"]
        n = 16
        rolls, pitches, schedule = [], [], []
        for i in range(n):
            r = math.radians(3.0 * math.sin(0.7 * i))
            p = math.radians(2.0 * math.cos(1.1 * i))
            rolls.append(r)
            pitches.append(p)
            slope, intercept = horizon_line_from_up(up_from_pitch_roll(p, r), K)
            conf = 0.9
            if i == 3:
                conf = 0.1            # below --min-confidence -> skipped
            if i == 4:
                intercept += 500.0    # beyond --max-tilt-px -> skipped
            schedule.append((slope, intercept, conf))
        # frame 6 has no IMU sample within the gap window -> att is None
        prefix = _make_quad(tmp_path, n, rolls, pitches, imu_skip={6})
        _stub_horizon(monkeypatch, schedule)

        plot = tmp_path / "out" / "check.png"
        rc = iac.main(["--triplet", str(prefix), "--plot", str(plot)])
        assert rc == 0
        out = capsys.readouterr().out
        assert f"{n} frames" in out
        # 16 - lowconf - tilt - imu gap = 13 pairs
        assert "13 paired with IMU samples" in out
        assert ("Best mapping: swap_roll_pitch=False "
                "sign_roll=+1 sign_pitch=+1") in out
        # horizon == mapped IMU exactly -> offsets ~0, residual ~0
        assert "swap_roll_pitch: false, sign_roll: 1, sign_pitch: 1" in out
        assert plot.exists() and plot.stat().st_size > 0
        assert f"wrote {plot}" in out

    def test_inverted_roll_sign_detected(self, tmp_path, monkeypatch, capsys):
        K = load_intrinsics()["fisheye"]["K"]
        n = 14
        rolls, pitches, schedule = [], [], []
        for i in range(n):
            r = math.radians(3.0 * math.sin(0.7 * i))
            p = math.radians(2.0 * math.cos(1.1 * i))
            rolls.append(-r)          # IMU roll axis flipped on the mount
            pitches.append(p)
            schedule.append(
                horizon_line_from_up(up_from_pitch_roll(p, r), K) + (0.9,))
        prefix = _make_quad(tmp_path, n, rolls, pitches)
        _stub_horizon(monkeypatch, schedule)
        rc = iac.main(["--triplet", str(prefix)])
        assert rc == 0
        out = capsys.readouterr().out
        assert ("Best mapping: swap_roll_pitch=False "
                "sign_roll=-1 sign_pitch=+1") in out

    def test_no_imu_sidecar_returns_1(self, tmp_path, capsys):
        scene = tmp_path / "2099-07-08"
        scene.mkdir()
        (scene / f"fisheye_{TS}.mp4").write_bytes(b"x")
        (scene / f"thermal_{TS}.mp4").write_bytes(b"x")
        (scene / f"mmwave_{TS}.csv").write_text("Date,Time,X,Y,Z\n")
        rc = iac.main(["--triplet", str(scene / TS)])
        assert rc == 1
        assert "no imu_" in capsys.readouterr().err

    def test_header_only_imu_returns_1(self, tmp_path, capsys):
        scene = tmp_path / "2099-07-08"
        scene.mkdir()
        (scene / f"fisheye_{TS}.mp4").write_bytes(b"x")
        (scene / f"thermal_{TS}.mp4").write_bytes(b"x")
        (scene / f"mmwave_{TS}.csv").write_text("Date,Time,X,Y,Z\n")
        (scene / f"imu_{TS}.csv").write_text("Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n")
        rc = iac.main(["--triplet", str(scene / TS)])
        assert rc == 1
        assert "empty" in capsys.readouterr().err

    def test_too_few_pairs_returns_1(self, tmp_path, monkeypatch, capsys):
        K = load_intrinsics()["fisheye"]["K"]
        cy = float(K[1, 2])
        n = 4                                       # < 10 pairs
        prefix = _make_quad(tmp_path, n, None, None)
        _stub_horizon(monkeypatch, [(0.0, cy, 0.9)] * n)
        rc = iac.main(["--triplet", str(prefix)])
        assert rc == 1
        assert "Too few pairs" in capsys.readouterr().err
