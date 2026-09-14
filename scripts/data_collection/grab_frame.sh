#!/usr/bin/env bash
# grab_frame.sh: save a single fisheye frame to results/.
#
#     bash scripts/data_collection/grab_frame.sh
#     bash scripts/data_collection/grab_frame.sh --out results/before_remount.jpg
#     bash scripts/data_collection/grab_frame.sh -n 5 --settle 4
#
# Run on the Pi, from anywhere (the script finds the repo itself).
#
# Captures at FISHEYE_SIZE (864x648), imported from the capture module, because
# that is the size configs/intrinsics.yaml is calibrated for: the frame can be
# undistorted or fed to the pipeline as-is. `rpicam-still` instead gives the
# full 2592x1944 sensor readout, which does NOT match the intrinsics and is the
# wrong input for any geometry work.
#
# The capture service owns the camera, so it has to be stopped first; this
# refuses to run rather than letting picamera2 fail with a device-busy trace.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=""
SETTLE=2
COUNT=1
FORCE=0

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'OPTS'

Options:
  --out FILE      output path (default results/fisheye_<timestamp>.jpg).
                  With -n > 1 an index is appended before the extension.
  --settle SECS   auto-exposure/white-balance settling time (default 2).
                  Below ~1 s the frame comes out dark or green-cast.
  -n COUNT        number of frames to grab (default 1).
  --force         capture even if the capture service is running.
  -h, --help      this text.
OPTS
}

while [ $# -gt 0 ]; do
    case "$1" in
        --out)    OUT="${2:?--out needs a path}"; shift 2 ;;
        --settle) SETTLE="${2:?--settle needs seconds}"; shift 2 ;;
        -n)       COUNT="${2:?-n needs a count}"; shift 2 ;;
        --force)  FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# The service holds the camera exclusively; picamera2 would fail obscurely.
if [ "$FORCE" -eq 0 ] && systemctl is-active --quiet asvproject-capture 2>/dev/null; then
    {
        echo "asvproject-capture is running and owns the camera. Either:"
        echo
        echo "  stop it first (remember to start it again afterwards):"
        echo "    sudo systemctl stop asvproject-capture"
        echo
        echo "  or re-run this with --force to try anyway."
    } >&2
    exit 1
fi

cd "$REPO"
mkdir -p results

OUT="$OUT" SETTLE="$SETTLE" COUNT="$COUNT" python3 - <<'PY'
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2

from scripts.data_collection.continuous_capture import FISHEYE_SIZE

out_arg = os.environ.get("OUT") or ""
settle = float(os.environ["SETTLE"])
count = int(os.environ["COUNT"])

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"format": "RGB888", "size": FISHEYE_SIZE}))
cam.start()
# Auto-exposure and white balance need a moment; without this the first frame
# is dark or green-cast, which has fooled more than one "the camera is broken".
time.sleep(settle)

try:
    for i in range(count):
        frame = cam.capture_array()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if out_arg:
            base, ext = os.path.splitext(out_arg)
            path = f"{base}_{i:02d}{ext or '.jpg'}" if count > 1 else out_arg
        else:
            suffix = f"_{i:02d}" if count > 1 else ""
            path = f"results/fisheye_{stamp}{suffix}.jpg"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not cv2.imwrite(path, frame):
            raise SystemExit(f"could not write {path}")
        h, w = frame.shape[:2]
        print(f"wrote {path}  {w}x{h}")
        if i + 1 < count:
            time.sleep(0.3)
finally:
    cam.stop()
    cam.close()
PY
