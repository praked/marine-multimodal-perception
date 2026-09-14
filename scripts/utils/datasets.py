"""Helpers for resolving and indexing the data/ triplet folders.

A "triplet" is the (fisheye mp4, thermal mp4, mmwave csv) tuple that
shares a timestamp suffix. The on-disk layout is:

    data/<scene>/fisheye_<ts>.mp4
    data/<scene>/thermal_<ts>.mp4
    data/<scene>/mmwave_<ts>.csv
    data/<scene>/imu_<ts>.csv        (optional, since 2026-07-08)
    data/<scene>/frames_<ts>.csv     (optional, since 2026-07-16)
    data/<scene>/gps_<ts>.csv        (optional, since 2026-08-24)

A *triplet prefix* (used by --triplet flags) is
`data/<scene>/<ts>`: the scene folder plus the bare timestamp.

Fisheye-only captures (the capture service run with ASVPROJECT_FISHEYE_ONLY=1)
have no thermal or mmwave file. They are resolved by passing
`require=("fisheye",)` to `resolve_triplet`; the default stays strict so the
fusion pipeline still refuses an incomplete clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"

SCENES = ("Boats", "Ducks", "OpenWater", "Rain")


def list_capture_missions() -> list[str]:
    """Return mission folder names under data/captures/, sorted."""
    if not CAPTURES_DIR.is_dir():
        return []
    return sorted(p.name for p in CAPTURES_DIR.iterdir() if p.is_dir())


@dataclass(frozen=True)
class Triplet:
    scene: str
    timestamp: str
    fisheye: Path
    # thermal / mmwave are None only for a fisheye-only capture, resolved with
    # `require=("fisheye",)`. Everything that runs the fusion pipeline resolves
    # strictly (the default) and therefore always sees real paths here.
    thermal: Path | None
    mmwave: Path | None
    # Optional 4th stream: BNO08x rotation-vector quaternions captured over
    # BLE alongside the box sensors (imu_<ts>.csv, since 2026-07-08).
    # None for clips without one; header-only files resolve but load empty.
    imu: Path | None = None
    # Optional per-frame timestamp sidecar (frames_<ts>.csv, since 2026-07-16):
    # frame_index -> RTC-backed wall-clock time. None for older clips, where the
    # frame time is reconstructed as chunk_start + index/FPS (see load_frames_csv).
    frames: Path | None = None
    # Optional own-boat position (gps_<ts>.csv, since 2026-08-24): read from
    # boat1's autopilot log at capture init and every ~30 min. Metadata, not a
    # sensor stream -- it is what gives a clip its true sun geometry instead of
    # the fixed lake position build_features falls back to.
    gps: Path | None = None

    @property
    def clip_id(self) -> str:
        return f"{self.scene}/{self.timestamp}"


def _resolve_video(scene_dir: Path, kind: str, ts: str) -> Path:
    """Pick the best on-disk copy of `<kind>_<ts>.mp4`.

    Field captures interrupted mid-chunk leave the primary mp4 without a moov
    atom; repaired copies land in `recovered/`, with `_trimmed` variants cut to
    the frames that align with the radar CSV. Preference order:
    recovered/*_trimmed.mp4 > recovered/*.mp4 > the primary file.
    """
    candidates = (
        scene_dir / "recovered" / f"{kind}_{ts}_trimmed.mp4",
        scene_dir / "recovered" / f"{kind}_{ts}.mp4",
        scene_dir / f"{kind}_{ts}.mp4",
    )
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]   # canonical path for the error message


#: The streams `resolve_triplet` insists on by default: the full sensor set
#: the fusion pipeline consumes.
CORE_STREAMS = ("fisheye", "thermal", "mmwave")


def resolve_triplet(prefix: str | Path,
                    require: tuple[str, ...] = CORE_STREAMS) -> Triplet:
    """Resolve `data/Boats/2025-06-23_16-21-07` -> a Triplet of files.

    Videos prefer recovered/trimmed copies when present (see _resolve_video).
    An `imu_<ts>.csv` sidecar is attached when it exists (None otherwise).

    `require` names the streams that must be on disk; anything not required and
    not present comes back as None. It defaults to all three, so the fusion
    pipeline and every multi-sensor tool keep failing loudly on an incomplete
    clip. Pass `require=("fisheye",)` in fisheye-only tools (undistortion,
    segmentation) so they also accept captures recorded with the radar and
    thermal camera disabled; see ASVPROJECT_FISHEYE_ONLY in
    scripts/data_collection/continuous_capture.py.
    """
    prefix = Path(prefix)
    scene = prefix.parent.name
    ts = prefix.name
    # Nested mission folders (data/captures/<mission>/<sub>/<ts>, e.g. the
    # 2026-08-19 afloat outing's session/ + calibration/ split) flatten to
    # `<mission>_<sub>` so frame_ids match the on-disk artefact naming that
    # data/seg, data/det and data/det_seg use (the gpu_corpus convention:
    # `2026-08-19_afloat_session__<ts>`). Single-level missions and the
    # canonical data/<Scene>/ layout are unaffected.
    try:
        rel = prefix.parent.resolve().relative_to(CAPTURES_DIR.resolve())
        if len(rel.parts) > 1:
            scene = "_".join(rel.parts)
    except (ValueError, OSError):
        pass
    fisheye = _resolve_video(prefix.parent, "fisheye", ts)
    thermal = _resolve_video(prefix.parent, "thermal", ts)
    mmwave = prefix.parent / f"mmwave_{ts}.csv"
    found = {"fisheye": fisheye, "thermal": thermal, "mmwave": mmwave}
    for name, p in found.items():
        if not p.exists():
            if name in require:
                raise FileNotFoundError(f"Missing {p}")
            found[name] = None
        elif name not in require and p.suffix == ".mp4":
            # A present-but-unreadable video (hard-power-cut / sensor-death
            # chunks: no moov atom) is honestly ABSENT for a non-required
            # stream — e.g. the 2026-08-19 post-16:28 thermal files.
            # Required streams still fail loudly downstream.
            import cv2
            cap = cv2.VideoCapture(str(p))
            ok = cap.isOpened()
            cap.release()
            if not ok:
                found[name] = None
    imu = prefix.parent / f"imu_{ts}.csv"
    frames = prefix.parent / f"frames_{ts}.csv"
    gps = prefix.parent / f"gps_{ts}.csv"
    return Triplet(scene=scene, timestamp=ts, fisheye=found["fisheye"],
                   thermal=found["thermal"], mmwave=found["mmwave"],
                   imu=imu if imu.exists() else None,
                   frames=frames if frames.exists() else None,
                   gps=gps if gps.exists() else None)


def list_triplets(scene: str | None = None,
                  respect_curation: bool = True) -> list[Triplet]:
    """Return every complete triplet under data/<scene>/ (or all scenes).

    Also enumerates `data/captures/<mission>/`: each mission folder is
    surfaced as its own scene (using the mission name verbatim) so field
    captures show up next to the canonical Boats/Ducks/OpenWater/Rain
    scenes in the dashboard. When `scene` is given, only that scene is
    returned; the lookup tries `data/<scene>/` first and falls back to
    `data/captures/<scene>/`.

    Sets deleted in the dashboard (configs/curation.yaml, exported from
    `sail_curation`; see scripts/utils/curation.py) are omitted by default;
    `respect_curation=False` lists everything on disk. Files are never
    touched: a deleted set is still there for `resolve_triplet`.
    """
    if scene is not None:
        candidate_dirs = [(scene, DATA_DIR / scene),
                          (scene, CAPTURES_DIR / scene)]
    else:
        candidate_dirs = [(s, DATA_DIR / s) for s in SCENES]
        candidate_dirs.extend((m, CAPTURES_DIR / m)
                              for m in list_capture_missions())
    out: list[Triplet] = []
    for s, scene_dir in candidate_dirs:
        if not scene_dir.is_dir():
            continue
        # Chunks directly in the mission folder, plus one level of session
        # sub-folders (per-boot `captures/<stamp>/` since 2026-08-25, and the
        # 2026-08-19 session/calibration split). `resolve_triplet` flattens a
        # nested prefix to the `<mission>_<sub>` scene, so nothing downstream
        # changes; a mission folder without sub-folders behaves as before.
        chunk_dirs = [scene_dir] + sorted(
            d for d in scene_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != "recovered"
            and any(d.glob("fisheye_*.mp4")))
        for cdir in chunk_dirs:
            for fish in sorted(cdir.glob("fisheye_*.mp4")):
                ts = fish.stem[len("fisheye_"):]
                try:
                    out.append(resolve_triplet(cdir / ts))
                except FileNotFoundError:
                    continue
    if respect_curation and out:
        from scripts.utils.curation import load_curation
        cur = load_curation()
        if len(cur):
            out = [t for t in out if not cur.is_deleted(t.clip_id)]
    return out


#: Physically possible bound (metres) for radar points. The AWR1843
#: profile in test_config.cfg gives max unambiguous range 9.02 m; 50 m
#: is generous slack to allow for any future profile changes while still
#: catching misaligned float32 reads from corrupted TLV packets (typical
#: garbage magnitudes are 1e30+).
MMWAVE_SANITY_BOUND_M = 50.0

#: Physically possible bound (m/s) for the per-point radial Doppler
#: velocity. The capture profile's max radial velocity is ±1 m/s
#: (test_config.cfg header); 10 m/s is generous slack for future profile
#: changes. Values beyond it are set to NaN (not row-dropped: the point's
#: position may still be fine: downstream falls back to the cross-frame
#: tracker for that point's velocity).
MMWAVE_DOPPLER_SANITY_MPS = 10.0


def load_mmwave_csv(path: str | Path,
                    sanity_bound_m: float = MMWAVE_SANITY_BOUND_M
                    ) -> pd.DataFrame:
    """Load a mmwave CSV and add a `RoundedTime` column (100 ms granularity).

    Two row shapes are accepted:

    - **Point rows**: `Date,Time,X,Y,Z[,V]` with finite X/Y/Z. Rows where
      any coordinate exceeds `sanity_bound_m` are silently dropped
      (TLV-parser garbage from the capture scripts, O(1e30+); CLAUDE.md §5).
    - **Sentinel rows**: `Date,Time,,,` with X/Y/Z empty (NaN). Written
      by `data_collection/continuous_capture.py` when the radar reported
      `n_obj == 0` (or `n_obj > 100`) so on-disk artefacts distinguish a
      live-but-empty radar from a dead one. The row carries no point
      cloud; the timestamp survives so iterate_triplet can use it as a
      heartbeat.

    The `V` column (per-point radial Doppler velocity, m/s; captured
    since 2026-07-06) is optional. Older CSVs without it load exactly as
    before; when present, implausible values (|V| >
    `MMWAVE_DOPPLER_SANITY_MPS`) are set to NaN so the point keeps its
    position but contributes no Doppler.

    The `SNR` / `NOISE` columns (per-point detection SNR and noise floor
    in dB, from the radar's TLV-7 side info; captured since 2026-07-09)
    are equally optional: coerced to numeric when present (empty →
    NaN), absent columns are simply absent. No sanity bound: the
    int16-in-0.1-dB wire format cannot produce out-of-range values the
    way spliced float TLVs could.
    """
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Date", "Time", "X", "Y", "Z"])
    if df.empty:
        df["Timestamp"] = pd.Series([], dtype="datetime64[ns]")
        df["RoundedTime"] = pd.Series([], dtype="object")
        return df
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["RoundedTime"] = df["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-5]
    if sanity_bound_m is not None:
        has_xyz = df[["X", "Y", "Z"]].notna().all(axis=1)
        out_of_bounds = df[["X", "Y", "Z"]].abs().ge(sanity_bound_m).any(axis=1)
        df = df[~(has_xyz & out_of_bounds)].reset_index(drop=True)
    if "V" in df.columns:
        v = pd.to_numeric(df["V"], errors="coerce")
        df["V"] = v.mask(v.abs() > MMWAVE_DOPPLER_SANITY_MPS)
    for col in ("SNR", "NOISE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_imu_csv(path: str | Path) -> pd.DataFrame:
    """Load an imu_<ts>.csv attitude sidecar. Two on-disk formats, auto-detected:

    * `Date,Time,Yaw,Pitch,Roll[,Ax,Ay,Az]`: native Euler from the directly-
      wired BNO085 in UART-RVC mode (default since 2026-07-14). Angles in
      degrees; optional acceleration in m/s^2.
    * `Date,Time,W,X,Y,Z`: legacy rotation-vector unit quaternion from the BLE
      bridge on boat1 (the 2026-07-08 quad clips).

    One row per capture loop when a fresh sample was available; gaps mean the
    stream stalled. An interrupted capture can leave a trailing all-NaN row
    (same failure mode as the radar CSV); those rows are dropped here.

    Adds `Timestamp` (datetime) and `RoundedTime` (HH:MM:SS.f, 100 ms: the same
    convention as load_mmwave_csv) so IMU samples can be matched to the
    radar-keyed frame timestamps. The value columns are preserved verbatim so
    downstream (imu_replay) can detect the format. Returns an empty frame (with
    Timestamp/RoundedTime) for missing/empty/header-only files so callers need
    no special-casing.
    """
    euler_cols = ["Yaw", "Pitch", "Roll"]
    quat_cols = ["W", "X", "Y", "Z"]
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df = pd.DataFrame()
    # Presence of the value columns decides the format (Euler preferred).
    if all(c in df.columns for c in euler_cols):
        value_cols = euler_cols
    elif all(c in df.columns for c in quat_cols):
        value_cols = quat_cols
    else:
        value_cols = []
    if value_cols and not df.empty:
        df = df.dropna(subset=value_cols).reset_index(drop=True)
    if not value_cols or df.empty:
        out = pd.DataFrame({c: pd.Series([], dtype="float64")
                            for c in (value_cols or euler_cols)})
        out["Timestamp"] = pd.Series([], dtype="datetime64[ns]")
        out["RoundedTime"] = pd.Series([], dtype="object")
        return out
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["RoundedTime"] = df["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-5]
    return df


def load_gps_csv(path: str | Path) -> pd.DataFrame:
    """Load a gps_<ts>.csv position sidecar -> the columns as written plus
    `Timestamp` and `RoundedTime` (100 ms, the convention every other loader
    uses), so a position can be matched to the frame timestamps.

    Columns: `Date,Time,Lat,Lon,Fix,Source,FixTime`. Date/Time are when the row
    was written by this box; `FixTime` is the source's own UTC timestamp, and
    the two differ by up to the ~30 min poll interval.

    `Source` matters as much as the coordinates: `boat_log` is a live fix from
    boat1's autopilot, `ble` is the bridge, and **`fallback` is the fixed
    position configured in configs/gps.yaml, not a measurement**. Anything
    deriving a per-clip quantity from the position (sun elevation, weather
    lookup) should say which it had.

    Returns an empty frame (with Timestamp/RoundedTime) for a missing, empty or
    header-only file, so callers need no special-casing.
    """
    cols = ["Lat", "Lon", "Fix", "Source", "FixTime"]
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df = pd.DataFrame()
    if not df.empty and {"Lat", "Lon"}.issubset(df.columns):
        df = df.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    else:
        df = pd.DataFrame()
    if df.empty:
        out = pd.DataFrame({c: pd.Series([], dtype="float64"
                                         if c in ("Lat", "Lon") else "object")
                            for c in cols})
        out["Timestamp"] = pd.Series([], dtype="datetime64[ns]")
        out["RoundedTime"] = pd.Series([], dtype="object")
        return out
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["RoundedTime"] = df["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-5]
    return df


#: Optional per-frame camera exposure columns in frames_<ts>.csv (mirrors
#: continuous_capture.FRAMES_EXPOSURE_COLUMNS; absent on pre-2026-08-28 clips).
FRAMES_EXPOSURE_COLUMNS = ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux")


def load_frames_csv(path: str | Path | None,
                    n_frames: int | None = None,
                    chunk_start: "pd.Timestamp | str | None" = None,
                    fps: float = 3.0) -> pd.DataFrame:
    """Load a frames_<ts>.csv per-frame timestamp sidecar → `frame_index`,
    `Timestamp`, `RoundedTime` (100 ms, same convention as the other loaders),
    so a video frame can be matched to radar/imu/gps by real wall-clock time.

    Written since 2026-07-16 (one row per capture loop). Absent on older clips
    (`path` is None or missing), in which case, if `n_frames` and `chunk_start`
    are given, the timestamps are **reconstructed as chunk_start + index/fps**
    (the legacy assumption). This is the single fallback path so downstream code
    can always call this and get a frame→time table without special-casing.
    Returns an empty frame if there's nothing to build from.
    """
    p = Path(path) if path is not None else None
    if p is not None and p.exists():
        try:
            df = pd.read_csv(p)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            df = pd.DataFrame()
        if not df.empty and "frame_index" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
            df["RoundedTime"] = df["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-5]
            # Exposure metadata (since 2026-08-28: ExposureTime us,
            # AnalogueGain, DigitalGain, Lux) rides along when the sidecar has
            # it; blanks become NaN. Older sidecars simply lack the columns.
            cols = ["frame_index", "Timestamp", "RoundedTime"]
            cols += [c for c in FRAMES_EXPOSURE_COLUMNS if c in df.columns]
            return df[cols]

    # Fallback: synthesize from chunk_start + index/fps (pre-sidecar clips).
    if n_frames is not None and chunk_start is not None and fps > 0:
        start = pd.to_datetime(chunk_start)
        idx = list(range(n_frames))
        ts = [start + pd.Timedelta(seconds=i / fps) for i in idx]
        out = pd.DataFrame({"frame_index": idx, "Timestamp": ts})
        out["RoundedTime"] = out["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-5]
        return out

    return pd.DataFrame({"frame_index": pd.Series([], dtype="int64"),
                         "Timestamp": pd.Series([], dtype="datetime64[ns]"),
                         "RoundedTime": pd.Series([], dtype="object")})


if __name__ == "__main__":
    triplets = list_triplets()
    print(f"Found {len(triplets)} triplets in data/:")
    for t in triplets:
        print(f"  {t.clip_id}")
