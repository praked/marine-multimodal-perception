"""Field-capture ingestion: Branch B.1.

Pulls a batch of captures from a source directory (e.g. an rsync target from
the Pi), validates integrity, tags them with mission metadata, and lands them
under `data/captures/<mission_id>/`.

A capture is keyed by its timestamp and consists of:

    fisheye_<ts>.mp4     REQUIRED: the capture is defined by it
    thermal_<ts>.mp4     optional: absent on a fisheye-only capture
    mmwave_<ts>.csv      optional: absent on a fisheye-only capture
    imu_<ts>.csv         optional: attitude sidecar
    frames_<ts>.csv      optional: per-frame RTC timestamps (since 2026-07-16)

Only the fisheye is mandatory: the capture service supports reduced sensor sets
(ASVPROJECT_FISHEYE_ONLY / ASVPROJECT_{RADAR,THERMAL}_ENABLE), and a fisheye-only
recording is a first-class capture, not a broken triplet. A stream that IS
present must be valid: a truncated video or a malformed radar CSV still fails
the capture, so "absent" and "broken" never get confused.

Usage:
    python -m scripts.data.ingest --source ~/pi_dump --mission 2026-06-15_morning
    python -m scripts.data.ingest --source ~/pi_dump --mission test --dry-run

Each ingested capture gets a sidecar JSON next to it:

    data/captures/<mission>/<ts>.metadata.json

with mission_id, ingest_timestamp, sensor config hash (if test_config.cfg
is in the source), which streams were present, and validation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from scripts.utils.datasets import REPO_ROOT

CAPTURES_DIR = REPO_ROOT / "data" / "captures"
EXPECTED_CSV_COLS = {"Date", "Time", "X", "Y", "Z"}


def _find_triplets_in_source(source: Path) -> list[str]:
    """Return the timestamps of every capture in `source`.

    Keyed on the fisheye, which is the one mandatory stream; the optional
    streams are discovered per-timestamp by `_streams`. A fisheye-only capture
    is therefore found and ingested rather than silently skipped.
    """
    return sorted(p.stem[len("fisheye_"):]
                  for p in source.glob("fisheye_*.mp4"))


def _streams(source: Path, ts: str) -> dict[str, Path]:
    """Map stream name -> path for the streams actually present for `ts`."""
    candidates = {
        "fisheye": source / f"fisheye_{ts}.mp4",
        "thermal": source / f"thermal_{ts}.mp4",
        "mmwave":  source / f"mmwave_{ts}.csv",
        "imu":     source / f"imu_{ts}.csv",
        "frames":  source / f"frames_{ts}.csv",
    }
    return {name: p for name, p in candidates.items() if p.exists()}


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _validate(source: Path, ts: str) -> tuple[bool, dict]:
    """Open each PRESENT file briefly to confirm it's not truncated/empty.

    An absent optional stream is recorded as `{"present": False}` and does not
    fail the capture; a present stream that will not open, has no frames, or
    lacks its expected columns does.
    """
    present = _streams(source, ts)
    report: dict = {"timestamp": ts,
                    "streams": sorted(present),
                    "files": {}}

    if "fisheye" not in present:
        report["files"]["fisheye"] = {"ok": False, "error": "missing"}
        return False, report

    for label in ("fisheye", "thermal"):
        p = present.get(label)
        if p is None:
            report["files"][label] = {"ok": True, "present": False}
            continue
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            report["files"][label] = {"ok": False, "error": "cannot open"}
            return False, report
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if n < 1:
            report["files"][label] = {"ok": False, "error": "no frames"}
            return False, report
        report["files"][label] = {"ok": True, "present": True,
                                  "n_frames": n, "size": [w, h]}

    mm_p = present.get("mmwave")
    if mm_p is None:
        report["files"]["mmwave"] = {"ok": True, "present": False}
    else:
        try:
            df = pd.read_csv(mm_p, nrows=5)
        except Exception as e:
            report["files"]["mmwave"] = {"ok": False, "error": str(e)}
            return False, report
        if not EXPECTED_CSV_COLS.issubset(df.columns):
            report["files"]["mmwave"] = {
                "ok": False,
                "error": f"missing columns; have {list(df.columns)}",
            }
            return False, report
        report["files"]["mmwave"] = {"ok": True, "present": True,
                                     "columns": list(df.columns)}

    # Sidecars (imu / frames) are carried through and reported, never gating:
    # they are metadata, and a capture with a stalled IMU link is still good
    # footage.
    for label in ("imu", "frames"):
        p = present.get(label)
        report["files"][label] = ({"ok": True, "present": False} if p is None
                                  else {"ok": True, "present": True})
    return True, report


def ingest_one(source: Path, dest: Path, ts: str,
               sensor_config_hash: str | None, dry_run: bool) -> dict:
    ok, report = _validate(source, ts)
    if not ok:
        return {"timestamp": ts, "ok": False, "report": report}

    present = _streams(source, ts)

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for src in present.values():
            shutil.copy2(src, dest / src.name)
        metadata = {
            "timestamp": ts,
            "mission_id": dest.name,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            "sensor_config_hash": sensor_config_hash,
            "streams": sorted(present),
            "validation": report,
            "source_paths": {name: str(p) for name, p in present.items()},
        }
        (dest / f"{ts}.metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"timestamp": ts, "ok": True, "report": report,
            "streams": sorted(present)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Directory containing the capture files.")
    ap.add_argument("--mission", required=True, help="Mission identifier; becomes the dest folder name.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate only; do not copy or write metadata.")
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_dir():
        raise SystemExit(f"source not a directory: {source}")
    dest = CAPTURES_DIR / args.mission
    if dest.exists() and not args.dry_run:
        print(f"warning: {dest} already exists; new captures will be added alongside")

    captures = _find_triplets_in_source(source)
    if not captures:
        raise SystemExit(f"no captures in {source} (no fisheye_*.mp4 found)")

    config_hash = _hash_file(source / "test_config.cfg")
    print(f"ingesting {len(captures)} captures from {source} -> {dest}")
    print(f"sensor_config_hash: {config_hash}")
    if args.dry_run:
        print("(dry-run: no files copied)")

    n_ok = 0
    for ts in captures:
        r = ingest_one(source, dest, ts, config_hash, args.dry_run)
        flag = "OK" if r["ok"] else "FAIL"
        # Name the streams so a fisheye-only ingest is visible at a glance
        # rather than looking like a full capture that lost two sensors.
        streams = "+".join(r.get("streams", [])) or "-"
        print(f"  {flag:4s}  {ts}  [{streams}]")
        n_ok += int(r["ok"])

    print(f"done: {n_ok}/{len(captures)} ingested cleanly")


if __name__ == "__main__":
    main()
