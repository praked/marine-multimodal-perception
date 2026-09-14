"""Dump UNDISTORTED fisheye frames for clips, to be segmented on the GPU node.

The segmentation masks must live in the same coordinate space the range
geometry uses: the *undistorted* fisheye image (cv_common.undistort_fisheye,
the `FisheyeResult.undistorted` of the pipeline). This script walks a triplet
exactly as the pipeline does (iterate_triplet applies clip_overrides, then we
undistort with the calibrated intrinsics) and writes one JPEG per kept frame,
named by the radar timestamp so it round-trips to a dashboard frame_id
(`_frame_id_for`).

Layout (per clip): <out>/<scene>__<triplet_ts>/ts=<HH-MM-SS.f>.jpg
The matching masks come back to data/seg/<scene>__<triplet_ts>/ts=<...>.png and
are read by scripts/utils/segmentation.SegProvider.

    python -m scripts.eval.export_undistorted_frames \
        --triplet data/Boats/2025-06-23_16-21-07 \
        --out data/seg_frames --every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import undistort_fisheye
from scripts.utils.datasets import resolve_triplet


def clip_dirname(scene: str, triplet_ts: str) -> str:
    """Filesystem-safe per-clip dir name (mirrors frame_id's scene/ts pair)."""
    return f"{scene}__{triplet_ts}"


def safe_ts(frame_ts: str) -> str:
    """Radar timestamp -> filename-safe token (matches _frame_id_for)."""
    return frame_ts.replace(":", "-")


def export_clip(triplet_prefix: str, out_root: Path, every: int) -> int:
    # Only the fisheye is read here, so fisheye-only captures (radar + thermal
    # disabled at capture time) export exactly like full ones.
    triplet = resolve_triplet(triplet_prefix, require=("fisheye",))
    intr = load_intrinsics()
    detection = load_detection()
    K, D = intr["fisheye"]["K"], intr["fisheye"]["D"]

    out_dir = out_root / clip_dirname(triplet.scene, triplet.timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for i, (ts, fisheye_frame, _thermal, _pts) in enumerate(
        iterate_triplet(triplet, detection)
    ):
        if fisheye_frame is None or (i % every):
            continue
        undist = undistort_fisheye(fisheye_frame, K, D)
        cv2.imwrite(str(out_dir / f"ts={safe_ts(ts)}.jpg"), undist)
        n += 1
    print(f"{triplet.clip_id}: {n} frames -> {out_dir}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--triplet", action="append", default=[],
                    help="triplet prefix (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("data/seg_frames"))
    ap.add_argument("--every", type=int, default=5, help="keep 1 frame in N")
    args = ap.parse_args(argv)

    if not args.triplet:
        ap.error("pass at least one --triplet")

    total = 0
    ok = 0
    failed: list[str] = []
    for t in args.triplet:
        # A corrupt/truncated clip (e.g. a recording cut off mid-capture, so the
        # mp4 has no moov atom) must not abort the whole batch.
        try:
            total += export_clip(t, args.out, args.every)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {t}: {exc}")
            failed.append(t)
    print(f"Total: {total} frames across {ok}/{len(args.triplet)} clip(s).")
    if failed:
        print(f"Failed clips ({len(failed)}): {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
