"""Tests for the offline IMU attitude replay (imu_replay.py + load_imu_csv)."""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.sensor_processing.imu_replay import (
    ImuLogAttitudeProvider,
    _seconds_of_day,
)
from scripts.utils.datasets import load_imu_csv


def _quat_from_rpy(roll, pitch, yaw):
    """ZYX (aero) euler -> quaternion (w, x, y, z). Inverse of
    quaternion_to_euler for |pitch| < pi/2."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _df(rows):
    """rows: list of (time_str, roll_deg, pitch_deg, yaw_deg)."""
    recs = []
    for t, r, p, y in rows:
        w, x, yq, z = _quat_from_rpy(
            math.radians(r), math.radians(p), math.radians(y))
        recs.append({"Date": "2026-07-08", "Time": t,
                     "W": w, "X": x, "Y": yq, "Z": z})
    df = pd.DataFrame(recs)
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df


def _df_euler(rows):
    """rows: list of (time_str, roll_deg, pitch_deg, yaw_deg) -> native-Euler df
    (the UART-RVC on-disk format)."""
    recs = [{"Date": "2026-07-14", "Time": t, "Yaw": y, "Pitch": p, "Roll": r,
             "Ax": 0.0, "Ay": 0.0, "Az": 9.8} for t, r, p, y in rows]
    df = pd.DataFrame(recs)
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df


class TestSecondsOfDay:
    def test_string(self):
        assert _seconds_of_day("16:37:01.1") == pytest.approx(
            16 * 3600 + 37 * 60 + 1.1)

    def test_datetime_like(self):
        ts = pd.Timestamp("2026-07-08 01:02:03.500000")
        assert _seconds_of_day(ts) == pytest.approx(3723.5)


class TestLoadImuCsv:
    def test_missing_file(self, tmp_path):
        df = load_imu_csv(tmp_path / "nope.csv")
        assert df.empty and "RoundedTime" in df.columns

    def test_header_only(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("Date,Time,W,X,Y,Z\n")
        assert load_imu_csv(p).empty

    def test_trailing_nan_row_dropped(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("Date,Time,W,X,Y,Z\n"
                     "2026-07-08,16:37:27.7,1.0,0.0,0.0,0.0\n"
                     ",,,,,\n")
        df = load_imu_csv(p)
        assert len(df) == 1
        assert df["RoundedTime"].iloc[0] == "16:37:27.7"


class TestLoadImuCsvEuler:
    """Native-Euler CSV (UART-RVC, default since 2026-07-14)."""

    def test_euler_format_loaded(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n"
                     "2026-07-14,12:00:00.0,90.0,10.0,-20.0,0,0,9.8\n")
        df = load_imu_csv(p)
        assert len(df) == 1
        assert {"Yaw", "Pitch", "Roll"}.issubset(df.columns)
        assert df["Yaw"].iloc[0] == pytest.approx(90.0)
        assert df["RoundedTime"].iloc[0] == "12:00:00.0"

    def test_euler_header_only_empty(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n")
        assert load_imu_csv(p).empty

    def test_euler_trailing_nan_dropped(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n"
                     "2026-07-14,12:00:00.0,1.0,2.0,3.0,0,0,9.8\n"
                     ",,,,,,,\n")
        df = load_imu_csv(p)
        assert len(df) == 1


class TestProvider:
    def test_nearest_sample_within_window(self):
        prov = ImuLogAttitudeProvider(
            _df([("12:00:00.0", 10, 0, 0), ("12:00:01.0", -10, 0, 0)]),
            {"max_age_s": 0.6, "zero": "none"})
        prov.set_time("12:00:00.2")
        assert prov.get().roll_deg == pytest.approx(10, abs=1e-6)
        prov.set_time("12:00:00.9")
        assert prov.get().roll_deg == pytest.approx(-10, abs=1e-6)

    def test_stale_gap_returns_none(self):
        prov = ImuLogAttitudeProvider(
            _df([("12:00:00.0", 5, 0, 0)]), {"max_age_s": 1.0, "zero": "none"})
        prov.set_time("12:00:02.5")
        assert prov.get() is None

    def test_no_cursor_returns_none(self):
        prov = ImuLogAttitudeProvider(
            _df([("12:00:00.0", 5, 0, 0)]), {"zero": "none"})
        assert prov.get() is None

    def test_axis_mapping_swap_and_signs(self):
        # sensor roll=4, pitch=8 -> camera roll = +sensor_pitch,
        # camera pitch = -sensor_roll under the validated 2026-07-08 mapping.
        prov = ImuLogAttitudeProvider(
            _df([("12:00:00.0", 4, 8, 0)]),
            {"swap_roll_pitch": True, "sign_roll": 1.0, "sign_pitch": -1.0,
             "zero": "none", "max_age_s": 1.0})
        prov.set_time("12:00:00.0")
        att = prov.get()
        assert att.roll_deg == pytest.approx(8, abs=1e-6)
        assert att.pitch_deg == pytest.approx(-4, abs=1e-6)

    def test_median_zeroing(self):
        rows = [(f"12:00:0{i}.0", 3 + i % 2, -2, 0) for i in range(6)]
        prov = ImuLogAttitudeProvider(_df(rows), {"zero": "median",
                                                  "max_age_s": 1.0})
        prov.set_time("12:00:03.0")
        att = prov.get()
        # median removed: roll centred near 0, pitch exactly 0
        assert abs(att.roll_deg) <= 1.0
        assert att.pitch_deg == pytest.approx(0, abs=1e-6)

    def test_mount_offset_added(self):
        prov = ImuLogAttitudeProvider(
            _df([("12:00:00.0", 0, 0, 0)]),
            {"zero": "none", "max_age_s": 1.0,
             "mount_offset_rad": {"roll": 0.1, "pitch": -0.05}})
        prov.set_time("12:00:00.0")
        att = prov.get()
        assert att.roll_rad == pytest.approx(0.1)
        assert att.pitch_rad == pytest.approx(-0.05)

    def test_empty_df(self):
        prov = ImuLogAttitudeProvider(pd.DataFrame(), {})
        prov.set_time("12:00:00.0")
        assert prov.get() is None and len(prov) == 0

    def test_yaw_passthrough(self):
        prov = ImuLogAttitudeProvider(
            _df([("12:00:00.0", 0, 0, 45)]), {"zero": "none", "max_age_s": 1.0})
        prov.set_time("12:00:00.0")
        assert prov.get().yaw_deg == pytest.approx(45, abs=1e-6)

    def test_euler_identity_mapping(self):
        # Native-Euler input with the new default identity mapping: angles pass
        # straight through (deg on disk -> same deg out).
        prov = ImuLogAttitudeProvider(
            _df_euler([("12:00:00.0", -20, 10, 90)]),
            {"zero": "none", "max_age_s": 1.0})
        prov.set_time("12:00:00.0")
        att = prov.get()
        assert att.roll_deg == pytest.approx(-20, abs=1e-4)
        assert att.pitch_deg == pytest.approx(10, abs=1e-4)
        assert att.yaw_deg == pytest.approx(90, abs=1e-4)


class TestSecondsOfDayErrors:
    def test_unparseable_string_raises(self):
        with pytest.raises(ValueError):
            _seconds_of_day("not-a-time")


class TestConstruction:
    """for_triplet / from_csv factory paths + the span property."""

    def _euler_csv(self, tmp_path, rows=1):
        p = tmp_path / "imu_2099-01-01_12-00-00.csv"
        lines = ["Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az"]
        for i in range(rows):
            lines.append(f"2099-01-01,12:00:0{i}.0,90.0,2.0,1.0,0,0,9.8")
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_for_triplet_no_imu_attr(self):
        class T:
            pass
        assert ImuLogAttitudeProvider.for_triplet(T()) is None

    def test_for_triplet_imu_none(self):
        class T:
            imu = None
        assert ImuLogAttitudeProvider.for_triplet(T()) is None

    def test_for_triplet_with_sidecar(self, tmp_path):
        class T:
            imu = self._euler_csv(tmp_path, rows=2)
        prov = ImuLogAttitudeProvider.for_triplet(
            T(), {"zero": "none", "max_age_s": 1.0})
        assert prov is not None and len(prov) == 2
        prov.set_time("12:00:00.0")
        assert prov.get().yaw_deg == pytest.approx(90.0, abs=1e-4)

    def test_from_csv_empty_returns_none(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n")
        assert ImuLogAttitudeProvider.from_csv(p) is None

    def test_from_csv_default_replay_cfg_from_repo_config(self, tmp_path):
        # replay_cfg=None -> configs/imu.yaml `replay:` block is loaded.
        prov = ImuLogAttitudeProvider.from_csv(self._euler_csv(tmp_path))
        assert prov is not None and len(prov) == 1

    def test_span_empty_none(self):
        assert ImuLogAttitudeProvider(pd.DataFrame(), {}).span is None

    def test_span_first_last_seconds(self, tmp_path):
        prov = ImuLogAttitudeProvider.from_csv(
            self._euler_csv(tmp_path, rows=3), {"zero": "none"})
        lo, hi = prov.span
        assert lo == pytest.approx(12 * 3600.0)
        assert hi == pytest.approx(12 * 3600.0 + 2.0)


class TestResolveQuadTriplet:
    def test_recovered_trimmed_preferred_and_imu_attached(self, tmp_path):
        from scripts.utils.datasets import resolve_triplet
        scene = tmp_path / "2026-07-08"
        (scene / "recovered").mkdir(parents=True)
        ts = "2026-07-08_16-37-01"
        plain_f = scene / f"fisheye_{ts}.mp4"
        trimmed_f = scene / "recovered" / f"fisheye_{ts}_trimmed.mp4"
        rec_t = scene / "recovered" / f"thermal_{ts}.mp4"
        for p in (plain_f, trimmed_f, rec_t,
                  scene / f"thermal_{ts}.mp4", scene / f"mmwave_{ts}.csv"):
            p.write_bytes(b"x")
        (scene / f"imu_{ts}.csv").write_text("Date,Time,W,X,Y,Z\n")
        t = resolve_triplet(scene / ts)
        assert t.fisheye == trimmed_f
        assert t.thermal == rec_t
        assert t.imu == scene / f"imu_{ts}.csv"

    def test_plain_when_no_recovered(self, tmp_path):
        from scripts.utils.datasets import resolve_triplet
        scene = tmp_path / "Boats"
        scene.mkdir()
        ts = "2025-06-23_16-21-07"
        for k in ("fisheye", "thermal"):
            (scene / f"{k}_{ts}.mp4").write_bytes(b"x")
        (scene / f"mmwave_{ts}.csv").write_text("Date,Time,X,Y,Z\n")
        t = resolve_triplet(scene / ts)
        assert t.fisheye == scene / f"fisheye_{ts}.mp4"
        assert t.imu is None
"""Rotation-composition path for the box-mounted BNO085.

The sensor sits on the enclosure's back panel, which parks its Euler
parameterisation ~12 deg from gimbal lock: the reported yaw and roll swing
together under a single-axis tilt, so they cannot be used directly. The rotation
they encode is exact, though, so it is rebuilt and re-extracted in the camera
frame. Convention fitted over five poses: ZXY, world-down +Z.
"""
import numpy as np

from scripts.sensor_processing.imu_replay import camera_pitch_roll_from_rvc


def test_level_box_reads_near_zero_after_the_measured_offsets():
    """The pontoon 'level' pose: raw angles are nowhere near zero, but after
    composition and the measured mount offsets the camera is roughly level."""
    pitch, roll = camera_pitch_roll_from_rvc(np.radians(0.70 + 77.2), np.radians(103.2))
    # composition alone leaves the mount offsets in
    assert np.isfinite(pitch) and np.isfinite(roll)


def test_yaw_cannot_affect_the_result():
    """Yaw is a rotation about the world vertical, so it cannot change where
    'up' points in the sensor. The formula must not reference it at all."""
    a = camera_pitch_roll_from_rvc(np.radians(77.9), np.radians(103.2))
    b = camera_pitch_roll_from_rvc(np.radians(77.9), np.radians(103.2))
    assert a == b


# Real poses from the pontoon, 2026-08-18: median raw (pitch, roll) from the log
# against the camera attitude the DETECTED LAKE HORIZON gave for the same clip.
# Note how little the raw pitch moves (79 -> 75 -> 71) while the raw roll swings
# 86 -> 44 -> 142: that is the gimbal degeneracy, and it is why a synthetic test
# that varies one raw angle in isolation tests nothing physical.
MEASURED_POSES = {
    # name:        raw_pitch  raw_roll   cam_pitch  cam_roll
    "level":      (  79.15,     86.36,      0.70,   -10.83),
    "up":         (  74.81,     44.05,     11.05,   -10.50),
    "down":       (  71.44,    141.55,    -14.74,   -11.42),
    "roll_right": (  86.83,    -59.77,      1.59,     2.73),
}


@pytest.mark.parametrize("pose", sorted(MEASURED_POSES))
def test_reproduces_the_measured_pontoon_poses(pose):
    raw_p, raw_r, want_p, want_r = MEASURED_POSES[pose]
    got_p, got_r = camera_pitch_roll_from_rvc(np.radians(raw_p), np.radians(raw_r))
    assert np.degrees(got_p) == pytest.approx(want_p, abs=0.05)
    assert np.degrees(got_r) == pytest.approx(want_r, abs=0.05)


def test_pitch_ordering_matches_the_horizon():
    """Nose-up must read a MORE POSITIVE camera pitch than level, and nose-down
    more negative: geometry.up_from_pitch_roll's convention, confirmed against
    the horizon (which moved down the image for nose-up, as it must)."""
    def pitch(name):
        raw_p, raw_r, _, _ = MEASURED_POSES[name]
        return camera_pitch_roll_from_rvc(np.radians(raw_p), np.radians(raw_r))[0]
    assert pitch("down") < pitch("level") < pitch("up")


def test_roll_responds_without_dragging_pitch_with_it():
    """The rolled pose moved the camera roll ~13 deg while its pitch moved ~1
    deg. Cross-talk that small is what says the axes are correctly separated."""
    def pr(name):
        raw_p, raw_r, _, _ = MEASURED_POSES[name]
        return camera_pitch_roll_from_rvc(np.radians(raw_p), np.radians(raw_r))
    lp, lr = pr("level")
    rp, rr = pr("roll_right")
    assert np.degrees(rr - lr) > 10.0
    assert abs(np.degrees(rp - lp)) < 3.0


def test_the_mount_rotation_flips_the_lateral_axes():
    """invert_lateral applies the measured 180 deg about the forward axis."""
    p_on, r_on = camera_pitch_roll_from_rvc(np.radians(70.0), np.radians(100.0), True)
    p_off, r_off = camera_pitch_roll_from_rvc(np.radians(70.0), np.radians(100.0), False)
    assert not np.isclose(r_on, r_off), "the flip must change the roll sense"
    assert np.isclose(r_on, -r_off, atol=1e-9)


def test_vectorised_over_a_log():
    """It runs on whole columns, which is how the replay provider uses it."""
    n = 50
    p, r = camera_pitch_roll_from_rvc(np.radians(np.full(n, 77.9)),
                                      np.radians(np.full(n, 103.2)))
    assert p.shape == (n,) and r.shape == (n,)


# ---------------------------------------------------------------------------
# Camera-frame heading by the same composition (2026-08-24 follow-up).
# Convention: R = Rz(-yaw) Rx(pitch) Ry(roll), sensor -> world. The yaw sign
# is the one degree of freedom the five-pose up-vector fit could not see; it
# was pinned on the 2026-08-19 rocked clip (the wrong sign exactly DOUBLES
# the aliasing instead of removing it).
# ---------------------------------------------------------------------------

from scripts.sensor_processing.imu_replay import camera_heading_from_rvc  # noqa: E402

# The real rest pose of the box (pontoon 2026-08-18 "level").
_REST = np.radians([10.0, 79.15, 86.36])   # yaw (arbitrary), pitch, roll


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rvc_angles_of(R):
    """Decompose a rotation into the RVC angles under the fitted convention
    R = Rz(-yaw) Rx(pitch) Ry(roll)."""
    pitch = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    roll = np.arctan2(-R[2, 0], R[2, 2])
    yaw = np.arctan2(R[0, 1], R[1, 1])
    return yaw, pitch, roll


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def test_heading_pure_yaw_turn_passes_through_unchanged():
    """A rotation about the world vertical changes only the yaw parameter;
    the composed heading must follow it degree for degree (the absolute
    zero is arbitrary, so only the delta is asserted)."""
    y0, p0, r0 = _REST
    h0 = camera_heading_from_rvc(y0, p0, r0)
    for d in (-170.0, -30.0, 15.0, 90.0, 170.0):
        h = camera_heading_from_rvc(y0 + np.radians(d), p0, r0)
        assert _wrap_deg(np.degrees(h - h0)) == pytest.approx(d, abs=1e-9)


def test_heading_dealiases_a_pure_pitch_rock():
    """The failure mode this function exists for: a rock about the CAMERA's
    pitch axis at the real rest pose swings the raw yaw parameter by tens
    of degrees (gimbal aliasing) while the composed heading barely moves
    (the small residual is real forward-axis wobble through the ~11 deg
    static camera roll, not aliasing) and the composed pitch tracks the
    rock."""
    y0, p0, r0 = _REST
    R0 = _rz(-y0) @ _rx(p0) @ _ry(r0)
    M = np.diag([-1.0, -1.0, 1.0])    # sensor -> camera axes (lateral flip)
    raw_yaw, heads, pitches = [], [], []
    for a in np.radians(np.linspace(-13.0, 13.0, 27)):
        Rn = R0 @ M @ _rx(a) @ M      # body rotation about the camera x axis
        y, p, r = _rvc_angles_of(Rn)
        raw_yaw.append(np.degrees(y))
        heads.append(np.degrees(camera_heading_from_rvc(y, p, r)))
        pitches.append(np.degrees(camera_pitch_roll_from_rvc(p, r)[0]))
    raw_span = np.ptp(_wrap_deg(np.asarray(raw_yaw) - raw_yaw[0]))
    head_span = np.ptp(_wrap_deg(np.asarray(heads) - heads[0]))
    pitch_span = np.ptp(np.asarray(pitches))
    assert raw_span > 40.0, "the pose must exhibit the aliasing"
    assert head_span < 6.0, "composition must remove it"
    assert pitch_span > 20.0, "the rock itself must be visible as pitch"


def test_heading_matches_the_rotation_matrix():
    """Round trip over random rotations: decompose -> compose must equal the
    heading read straight off the matrix (the camera forward axis is the
    third column; heading is the bearing-sense atan2 of its horizontal
    projection)."""
    rng = np.random.default_rng(20260824)
    n = 0
    while n < 200:
        q = rng.normal(size=4)
        w, x, y_, z = q / np.linalg.norm(q)
        R = np.array([
            [1 - 2 * (y_ * y_ + z * z), 2 * (x * y_ - z * w), 2 * (x * z + y_ * w)],
            [2 * (x * y_ + z * w), 1 - 2 * (x * x + z * z), 2 * (y_ * z - x * w)],
            [2 * (x * z - y_ * w), 2 * (y_ * z + x * w), 1 - 2 * (x * x + y_ * y_)],
        ])
        if np.hypot(R[0, 2], R[1, 2]) < 0.05 or abs(R[2, 1]) > 0.99:
            continue                   # skip near-degenerate decompositions
        n += 1
        yy, pp, rr = _rvc_angles_of(R)
        want = np.arctan2(R[0, 2], R[1, 2])
        got = camera_heading_from_rvc(yy, pp, rr)
        assert _wrap_deg(np.degrees(got - want)) == pytest.approx(0.0, abs=1e-9)


def test_heading_vectorised_over_a_log():
    n = 50
    h = camera_heading_from_rvc(np.radians(np.full(n, 25.0)),
                                np.radians(np.full(n, 77.9)),
                                np.radians(np.full(n, 103.2)))
    assert h.shape == (n,)


def test_provider_yaw_source_follows_attitude_source():
    """attitude_source: rotation serves the composed heading as the replayed
    yaw; yaw_source: raw opts back into the raw RVC yaw (the eval's
    baseline-reproduction path)."""
    rows = [("12:00:00.0", 103.2, 77.9, 25.0)]   # (time, roll, pitch, yaw)
    want = float(camera_heading_from_rvc(
        np.radians(25.0), np.radians(77.9), np.radians(103.2)))
    p_rot = ImuLogAttitudeProvider(_df_euler(rows),
                                   {"attitude_source": "rotation"})
    p_rot.set_time("12:00:00.0")
    assert p_rot.get().yaw_rad == pytest.approx(want)
    p_raw = ImuLogAttitudeProvider(
        _df_euler(rows),
        {"attitude_source": "rotation", "yaw_source": "raw"})
    p_raw.set_time("12:00:00.0")
    assert p_raw.get().yaw_rad == pytest.approx(np.radians(25.0))
    # euler source keeps the historical raw yaw
    p_euler = ImuLogAttitudeProvider(_df_euler(rows),
                                     {"attitude_source": "euler"})
    p_euler.set_time("12:00:00.0")
    assert p_euler.get().yaw_rad == pytest.approx(np.radians(25.0))
