"""Offline IMU attitude replay: the 4th stream of a captured quad clip.

`continuous_capture.py` writes an `imu_<ts>.csv` sidecar per clip. Two formats
are supported transparently (see datasets.load_imu_csv): native Euler
`Yaw,Pitch,Roll` from the directly-wired BNO085 in UART-RVC mode (default since
2026-07-14) and the legacy `W,X,Y,Z` rotation-vector quaternion from the boat1
BLE bridge (the 2026-07-08 quad clips). This module turns either log into an
attitude *provider* with the same `.get() -> Attitude | None` contract the live
`BNO085Reader` exposes, so the canonical
`ObstacleDetectionPipeline` consumes recorded IMU exactly like a live one:

    provider = ImuLogAttitudeProvider.for_triplet(triplet)
    pipeline = ObstacleDetectionPipeline(intr, det, attitude_provider=provider)
    for ts, fish, therm, mm in iterate_triplet(triplet, det):
        provider.set_time(ts)                # move the replay cursor
        res = pipeline.process_frame(fish, therm, mm, timestamp=ts)

The pipeline polls `.get()` with no notion of time, so the driver loop *must*
call `set_time(frame_ts)` before each frame; `get()` then returns the log
sample nearest the cursor, or None when the nearest sample is further than
`max_age_s` away (BLE gap / IMU connected late): the pipeline falls back to
the detected horizon / level exactly as it would on a live dropout.

Axis mapping: the raw (roll, pitch, yaw) are in the *sensor* frame. The
`replay:` block in configs/imu.yaml maps them onto the CAMERA's (pitch, roll)
via swap/sign knobs plus a fixed mount offset or clip-median zeroing. The old
boat1 BLE mount needed a ~90 deg swap; the new box-mounted UART-RVC IMU is
roughly aligned with the camera, so the default is now IDENTITY (no swap,
sign +1), but it is UNVALIDATED for this mount. Re-run
`scripts.eval.imu_attitude_check` on a fresh clip before trusting pitch/roll.

Under `attitude_source: rotation` (the box-mount default) pitch, roll AND yaw
are all re-derived from the full rotation the RVC angles encode
(`camera_pitch_roll_from_rvc` / `camera_heading_from_rvc`): the raw angles are
individually gimbal-aliased on this mount. `yaw_source: raw` opts the yaw back
to the raw sensor value (baseline reproduction in target_motion_eval).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.sensor_processing.imu_bno085 import (
    Attitude,
    load_imu_config,
    quaternion_to_euler,
)
from scripts.utils.datasets import load_imu_csv


def _seconds_of_day(ts: Any) -> float:
    """HH:MM:SS[.f] string / datetime / pandas Timestamp -> seconds of day."""
    if hasattr(ts, "hour"):
        return (ts.hour * 3600.0 + ts.minute * 60.0 + ts.second
                + getattr(ts, "microsecond", 0) / 1e6)
    parts = str(ts).split(":")
    if len(parts) != 3:
        raise ValueError(f"unparseable timestamp: {ts!r}")
    return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])


def camera_pitch_roll_from_rvc(pitch_rad, roll_rad, invert_lateral: bool = True):
    """Camera-frame (pitch, roll) from the RVC Euler angles, by composition.

    The BNO085 sits on the enclosure's back panel, which puts its Euler
    parameterisation about 12 degrees from gimbal lock: a single-axis tilt swings
    the reported yaw and roll together by ~79 degrees (measured 2026-08-17). The
    individual angles are therefore useless on their own, but the rotation they
    encode is exact, reproducing measured gravity to 0.01 degrees. So rebuild the
    rotation and re-extract in the camera frame, where the parameterisation is
    well conditioned.

    Convention fitted over five poses: order ZXY, world-down +Z. Under it the
    sensor-frame up vector is the third row of R,

        up_sensor = (-cos(p) sin(r),  sin(p),  cos(p) cos(r))

    which is independent of yaw, as it must be: yaw is a rotation about the
    world vertical and cannot change where "up" points in the sensor.

    `invert_lateral` applies the measured mount rotation, 180 degrees about the
    camera's forward axis (sensor X -> -X, Y -> -Y, Z -> +Z), giving

        up_cam = (cos(p) sin(r), -sin(p), cos(p) cos(r))

    Extraction then matches `geometry.up_from_pitch_roll`, i.e. pitch about
    camera X with nose-up positive and roll about camera Z.

    Validated against a lake horizon on 2026-08-18: camera and IMU track each
    other to ~1 degree over +-15 degree tilts in both axes, cross-talk included.
    """
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    ux, uy, uz = -cp * sr, sp, cp * cr
    if invert_lateral:
        ux, uy = -ux, -uy
    pitch = np.arctan2(uz, -uy)
    roll = np.arcsin(np.clip(-ux, -1.0, 1.0))
    return pitch, roll


def camera_heading_from_rvc(yaw_rad, pitch_rad, roll_rad):
    """Camera-frame heading from the RVC Euler angles, by composition.

    The raw RVC yaw is unusable on this mount for the same reason the raw
    pitch/roll are (see `camera_pitch_roll_from_rvc`): the Euler
    parameterisation sits ~12 degrees from gimbal lock, so pitch/roll ROCKING
    bleeds into the reported yaw by tens of degrees (measured 2026-08-24,
    docs/history/2026-08-24_target_motion.md section 5.3: the "correction" it fed
    injected fake crossings). The rotation the three angles encode is exact,
    though, so rebuild it and take the heading of the camera's FORWARD axis
    about the world vertical, which is exactly the angle that sweeps radar
    bearings when the boat turns.

    The full convention is ZXY with the yaw entering NEGATED:
    R = Rz(-yaw) Rx(pitch) Ry(roll), sensor -> world. The pitch/roll part is
    the 2026-08-17 five-pose fit; the yaw sign is a degree of freedom that
    fit could NOT see (the up vector is yaw-independent) and was pinned on
    the 2026-08-19 rocked clip: composed with Rz(-yaw) the +-13 deg pitch
    rock leaves heading std 3.9 deg (vs 36.9 raw, and 70.8 -- exactly
    doubled, the sign-error signature -- with Rz(+yaw)). The camera's
    forward axis is the sensor's +Z (the 180-degree lateral mount flip
    cannot move it), whose world direction is the third column of R:

        f_world = (cy sr - sy sp cr,  -sy sr - cy sp cr,  cp cr)

    Its vertical component cp*cr equals sin(camera pitch) -- near zero on
    this mount, so the horizontal projection is long and the extraction is
    well conditioned exactly where the Euler yaw is worst. Heading is the
    atan2 of that projection, in BEARING sense (right/clockwise-from-above
    positive, matching atan2(X, Y) radar bearings): a pure rotation about
    the world vertical that raises the reported RVC yaw by delta raises the
    returned heading by delta, so under it a world-fixed target's bearing
    is stabilised by theta_world = theta_boat + yaw (`imu_yaw_sign` +1,
    arbitrated by `target_motion_eval`). The absolute zero is arbitrary;
    only differences are meaningful.
    """
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    fx = cy * sr - sy * sp * cr
    fy = -sy * sr - cy * sp * cr
    return np.arctan2(fx, fy)


class ImuLogAttitudeProvider:
    """Replay a recorded quaternion log as a pipeline attitude provider.

    Duck-types the live `BNO085Reader` polling interface (`.get()`), plus a
    `set_time()` cursor the offline driver advances per frame. Stateless in
    between: `get()` may be called any number of times per frame.
    """

    def __init__(self, df: pd.DataFrame, replay_cfg: dict[str, Any] | None = None):
        cfg = dict(replay_cfg or {})
        self.max_age_s = float(cfg.get("max_age_s", 1.0))
        self._swap = bool(cfg.get("swap_roll_pitch", False))
        self._sign_roll = float(cfg.get("sign_roll", 1.0))
        self._sign_pitch = float(cfg.get("sign_pitch", 1.0))
        mount = cfg.get("mount_offset_rad", {}) or {}
        self._mount_pitch = float(mount.get("pitch", 0.0))
        self._mount_roll = float(mount.get("roll", 0.0))
        self._zero = str(cfg.get("zero", "none"))
        # "euler" (default) keeps the historical sign/swap/offset handling, which
        # is correct for mounts whose Euler parameterisation is well conditioned
        # (the boat1 BLE clips). "rotation" rebuilds the rotation and re-extracts
        # in the camera frame: required for the back-panel box mount, where the
        # raw angles sit beside gimbal lock. See camera_pitch_roll_from_rvc.
        self._attitude_source = str(cfg.get("attitude_source", "euler"))
        self._invert_lateral = bool(cfg.get("mount_invert_lateral", True))
        # Yaw follows the attitude source by default: under "rotation" the
        # replayed yaw is the composed camera-frame heading (the raw RVC yaw
        # is gimbal-aliased on the box mount, measured 2026-08-24 section 5.3);
        # under "euler" it stays the raw sensor yaw. Override with
        # yaw_source: raw|rotation (the eval uses raw to reproduce baselines).
        self._yaw_source = str(cfg.get("yaw_source", self._attitude_source))

        self._t = np.empty(0)
        self._roll = np.empty(0)
        self._pitch = np.empty(0)
        self._yaw = np.empty(0)
        self._cursor: float | None = None

        if df is not None and not df.empty:
            if all(c in df.columns for c in ("Yaw", "Pitch", "Roll")):
                # Native Euler (UART-RVC, default): degrees on disk. RVC already
                # emits fused Tait-Bryan angles, so no quaternion step.
                raw_roll = np.radians(df["Roll"].to_numpy(float))
                raw_pitch = np.radians(df["Pitch"].to_numpy(float))
                yaw = np.radians(df["Yaw"].to_numpy(float))
            else:
                # Legacy rotation-vector quaternion (BLE / boat1 clips).
                rpy = np.array([
                    quaternion_to_euler(w, x, y, z)
                    for w, x, y, z in df[["W", "X", "Y", "Z"]].to_numpy(float)
                ])
                raw_roll, raw_pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
            if self._yaw_source == "rotation":
                yaw = camera_heading_from_rvc(yaw, raw_pitch, raw_roll)
            if self._attitude_source == "rotation":
                pitch, roll = camera_pitch_roll_from_rvc(
                    raw_pitch, raw_roll, invert_lateral=self._invert_lateral)
            else:
                if self._swap:
                    raw_roll, raw_pitch = raw_pitch, raw_roll
                roll = self._sign_roll * raw_roll
                pitch = self._sign_pitch * raw_pitch
            if self._zero == "median":
                # Clip-median = level. Robust to the BNO08x mounting offset on
                # boat1 (unmeasured) as long as the boat spent most of the clip
                # near its rest attitude. A persistent heel would be zeroed
                # away too: prefer a measured mount_offset_rad when available.
                roll = roll - float(np.median(roll))
                pitch = pitch - float(np.median(pitch))
            self._t = np.array([_seconds_of_day(ts) for ts in df["Timestamp"]])
            self._roll = roll + self._mount_roll
            self._pitch = pitch + self._mount_pitch
            self._yaw = yaw

    # -- construction ---------------------------------------------------------

    @classmethod
    def for_triplet(cls, triplet, replay_cfg: dict[str, Any] | None = None
                    ) -> "ImuLogAttitudeProvider | None":
        """Provider for a Triplet's imu sidecar, or None when there is no
        usable log (no file, or header-only/empty)."""
        if getattr(triplet, "imu", None) is None:
            return None
        return cls.from_csv(triplet.imu, replay_cfg)

    @classmethod
    def from_csv(cls, path: str | Path, replay_cfg: dict[str, Any] | None = None
                 ) -> "ImuLogAttitudeProvider | None":
        df = load_imu_csv(path)
        if df.empty:
            return None
        if replay_cfg is None:
            replay_cfg = (load_imu_config().get("replay", {}) or {})
        return cls(df, replay_cfg)

    # -- replay interface ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._t)

    @property
    def span(self) -> tuple[float, float] | None:
        """(first, last) sample time in seconds-of-day, or None when empty."""
        if not len(self._t):
            return None
        return float(self._t[0]), float(self._t[-1])

    def set_time(self, ts: Any) -> None:
        """Advance the replay cursor to a frame timestamp (HH:MM:SS.f string,
        datetime, or seconds-of-day float)."""
        self._cursor = float(ts) if isinstance(ts, (int, float)) else _seconds_of_day(ts)

    def get(self) -> Attitude | None:
        """Attitude at the cursor: nearest log sample within max_age_s."""
        if self._cursor is None or not len(self._t):
            return None
        i = int(np.searchsorted(self._t, self._cursor))
        best, gap = None, math.inf
        for j in (i - 1, i):
            if 0 <= j < len(self._t):
                d = abs(self._t[j] - self._cursor)
                if d < gap:
                    best, gap = j, d
        if best is None or gap > self.max_age_s:
            return None
        return Attitude(
            pitch_rad=float(self._pitch[best]),
            roll_rad=float(self._roll[best]),
            yaw_rad=float(self._yaw[best]),
            timestamp=None,       # replay: never stale by wall-clock
            source="bno085:log",
        )
