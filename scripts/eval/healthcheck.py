"""One-command pre-trip sanity check.

Loads every config, lists every triplet, runs the canonical pipeline on
one frame of each clip, and prints a one-page status summary. Exits
nonzero if anything fails. Use this before getting on the plane.

Usage:
    python -m scripts.eval.healthcheck
    python -m scripts.eval.healthcheck --skip-tests   # skip the pytest run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import (
    DEFAULT_DETECTION,
    DEFAULT_INTRINSICS,
    load_detection,
    load_intrinsics,
)
from scripts.utils.datasets import REPO_ROOT, list_triplets
from scripts.utils.geometry import EXTRINSICS_PATH, load_extrinsics


def check(label: str, fn):
    """Run fn() and report OK / FAIL. Returns (ok, detail)."""
    try:
        detail = fn()
        print(f"  [OK]   {label}: {detail}")
        return True, detail
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc(limit=2)
        return False, str(e)


def section(title: str):
    print(f"\n=== {title} ===")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-tests", action="store_true",
                    help="Skip the pytest invocation (faster).")
    args = ap.parse_args()

    failures = 0

    section("configs")
    failures += not check("intrinsics.yaml", load_intrinsics)[0]
    intr = load_intrinsics()
    failures += not check("intrinsics.yaml exists", lambda: f"fisheye K shape {intr['fisheye']['K'].shape}")[0]
    det = load_detection()
    failures += not check("detection.yaml exists", lambda: f"{len(det)} sections")[0]
    extr = load_extrinsics()
    failures += not check("extrinsics.yaml exists",
                          lambda: f"measured={extr['measured']}")[0]
    if not extr["measured"]:
        print("         ! extrinsics are placeholders; A.1 overlay will be approximate")

    section("data")
    triplets = list_triplets()
    failures += not check("data/ triplets", lambda: f"{len(triplets)} found across {len({t.scene for t in triplets})} scenes")[0]

    section("pipeline smoke (one frame per clip)")
    smoke_failures = 0
    for tri in triplets:
        try:
            pl = ObstacleDetectionPipeline(intr, det)
            n = 0
            for ts, fish, therm, mm_pts in iterate_triplet(tri, det):
                pl.process_frame(fish, therm, mm_pts, timestamp=ts)
                n += 1
                if n >= 1:
                    break
            print(f"  [OK]   {tri.clip_id}")
        except Exception as e:
            print(f"  [FAIL] {tri.clip_id}: {e}")
            smoke_failures += 1
    failures += smoke_failures
    if smoke_failures == 0:
        print(f"  all {len(triplets)} triplets process at least one frame")

    section("test suite")
    if args.skip_tests:
        print("  skipped")
    else:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "--quiet",
                 "--no-header", "-x", "--tb=short"],
                cwd=str(REPO_ROOT),
                capture_output=True, text=True, timeout=600,
            )
            last_line = (res.stdout.splitlines() or [""])[-1]
            if res.returncode == 0:
                print(f"  [OK]   pytest: {last_line}")
            else:
                print(f"  [FAIL] pytest: {last_line}")
                print(res.stdout[-1000:])
                failures += 1
        except Exception as e:
            print(f"  [FAIL] could not run pytest: {e}")
            failures += 1

    section("summary")
    if failures == 0:
        print("  HEALTHY: pre-trip checks all passed")
        sys.exit(0)
    print(f"  {failures} check(s) failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
