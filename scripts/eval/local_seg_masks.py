"""Generate water-segmentation masks for a clip locally (ONNX, no GPU).

The offline masks under data/seg/ were produced on the InstTwo GPU node; for new
field captures we don't want to wait for a GPU booking. This runs the
distilled LRASPP student (models/onnx/, fp32 by default: 99.5% teacher
agreement, docs/history/2026-07-06_pi_quantisation.md §2026-07-07) over a clip's
UNDISTORTED fisheye frames on the laptop CPU and writes label-encoded PNGs
(0=obstacle, 1=water, 2=sky) in exactly the SegProvider layout:

    data/seg/<scene>__<triplet_ts>/ts=<HH-MM-SS.f>.png    (colons -> dashes)

Frame timestamps come from iterate_triplet (radar-keyed), so the mask names
line up 1:1 with the frame_ids fusion.py / the dashboard look up.

    python -m scripts.eval.local_seg_masks \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import undistort_fisheye
from scripts.utils.datasets import REPO_ROOT, resolve_triplet

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_ONNX = REPO_ROOT / "models" / "onnx" / "student864t_864x648.onnx"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--seg-root", default=str(REPO_ROOT / "data" / "seg"))
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate masks that already exist.")
    ap.add_argument("--min-luma", type=float, default=25.0,
                    help="Skip frames darker than this mean luminance (the "
                         "SegWorker darkness gate: the model fails "
                         "dishonestly on near-black frames).")
    args = ap.parse_args(argv)

    try:
        import onnxruntime as ort
    except ImportError:
        print("needs onnxruntime: pip install onnxruntime", file=sys.stderr)
        return 1

    # Segmentation runs on the fisheye alone, so a fisheye-only capture
    # (ASVPROJECT_FISHEYE_ONLY at capture time) is fully supported here.
    triplet = resolve_triplet(args.triplet, require=("fisheye",))
    intrinsics = load_intrinsics()
    detection = load_detection()
    intr = intrinsics["fisheye"]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    mw, mh = int(inp.shape[3]), int(inp.shape[2])
    print(f"model {Path(args.onnx).name} input {mw}x{mh}")

    out_dir = Path(args.seg_root) / f"{triplet.scene}__{triplet.timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = n_written = n_skipped_dark = 0
    t0 = time.monotonic()
    for ts, fish, _therm, _pts in iterate_triplet(triplet, detection):
        n += 1
        out_path = out_dir / f"ts={ts.replace(':', '-')}.png"
        if out_path.exists() and not args.overwrite:
            continue
        und = undistort_fisheye(fish, intr["K"], intr["D"])
        rgb = und[:, :, ::-1]                       # BGR -> RGB
        if float(rgb.mean()) < args.min_luma:
            n_skipped_dark += 1
            continue
        im = Image.fromarray(rgb)
        oh, ow = und.shape[:2]
        arr = (np.asarray(im.resize((mw, mh)), np.float32) / 255.0 - _MEAN) / _STD
        x = arr.transpose(2, 0, 1)[None].astype(np.float32)
        logits = sess.run(None, {inp.name: x})[0][0]
        cls = logits.argmax(0).astype(np.uint8)
        mask = Image.fromarray(cls).resize((ow, oh), Image.NEAREST)
        mask.save(out_path)
        n_written += 1
        if n_written % 50 == 0:
            rate = n_written / (time.monotonic() - t0)
            print(f"  {n_written} masks ({rate:.1f} fps)", flush=True)

    dt = time.monotonic() - t0
    print(f"done: {n} frames -> {n_written} masks written "
          f"({n_skipped_dark} skipped dark) in {dt:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
