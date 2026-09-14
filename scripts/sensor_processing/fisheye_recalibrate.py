"""Fisheye intrinsics recalibration (Kannala-Brandt), tuned for range accuracy.

A NEW tool: the original `fisheye_calibrate.py` (AuthorZero's) is left untouched.
It exists because the inlined calibration is under-constrained for monocular
range: range is computed from the bbox bottom, which sits low in the frame =
the high-distortion fisheye edge, and the previous 3x3-corner / centre-only
calibration left that region unconstrained (the undistortion extrapolates there,
moving bottom-corner pixels by 400+ px). What this adds:

  - configurable board (default 10x7 INNER corners = docs/checkered_board_10x7.pdf,
    which is 11x8 squares), denser than the 3x3 original;
  - reprojection error, overall + per image, as a quality gate (aim < ~0.5 px);
  - a corner COVERAGE map + report so you can CONFIRM the board reached the frame
    edges/corners: the gap that made range unreliable near the bottom;
  - saves K / D / image_size + the error to a YAML file and prints a paste-ready
    block for configs/intrinsics.yaml (it does NOT overwrite that file, to keep
    its comments and let you validate first with range_validate.py).

Capture protocol (on land): 20-40 shots of a rigid, flat 10x7 board at varied
angles/distances, and DELIBERATELY place it at the frame edges and especially
the BOTTOM corners, not just the centre; that's where near-object range needs
the constraint.

Usage:
  python -m scripts.sensor_processing.fisheye_recalibrate \
      --images 'data/fisheye_checkerboard/*.jpg' --cols 10 --rows 7 \
      --square-mm 25 --out configs/intrinsics_fisheye_recalib.yaml \
      --coverage-png results/screens/calib_coverage.png
"""

from __future__ import annotations

import argparse
import glob
import types
from pathlib import Path

import cv2
import numpy as np


def _find_corners(images: list[str], cols: int, rows: int):
    """Detect inner-corner grids; return (objpoints, imgpoints, size, used, skipped)."""
    pattern = (cols, rows)
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    # NB no CALIB_CB_FAST_CHECK: it is a cheap early-reject that discards
    # valid boards, badly, when the board is small in the frame or near an
    # edge, which is exactly where fisheye calibration needs the views.
    # Measured 2026-08-19 on a 45-image set: 1/8 detected with FAST_CHECK,
    # 22/45 without.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    size = None
    used: list[str] = []
    skipped: list[str] = []
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            skipped.append(fname)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]  # (w, h)
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            skipped.append(fname)
            continue
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix)
        objpoints.append(objp.copy())
        # cv2.fisheye.calibrate wants (1, N, 2) float; findChessboardCorners
        # returns (N, 1, 2). OpenCV 5 enforces this and rejects the latter with
        # an opaque "imagePoints.type() == CV_32FC2" assertion, which the
        # handler below used to misreport as "board views too similar".
        imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
        used.append(fname)
    return objpoints, imgpoints, size, used, skipped


def _coverage_report(imgpoints: list[np.ndarray], size, grid: int = 3) -> str:
    """3x3 occupancy of detected corners: flags unconstrained regions."""
    w, h = size
    counts = np.zeros((grid, grid), dtype=int)
    for pts in imgpoints:
        for (x, y) in pts.reshape(-1, 2):
            c = min(grid - 1, int(x / w * grid))
            r = min(grid - 1, int(y / h * grid))
            counts[r, c] += 1
    lines = ["corner coverage (rows = top->bottom, cols = left->right):"]
    for r in range(grid):
        lines.append("  " + " ".join(f"{counts[r, c]:5d}" for c in range(grid)))
    empty = [(r, c) for r in range(grid) for c in range(grid) if counts[r, c] == 0]
    if empty:
        names = {0: "top", 1: "mid", 2: "bottom"}
        cnames = {0: "left", 1: "centre", 2: "right"}
        where = ", ".join(f"{names[r]}-{cnames[c]}" for r, c in empty)
        lines.append(f"  WARNING: no corners in: {where}.")
        if any(r == grid - 1 for r, _ in empty):
            lines.append("  ** bottom row underconstrained -> NEAR-FIELD RANGE will "
                         "stay unreliable. Re-shoot with the board at the frame bottom. **")
    else:
        lines.append("  OK: all regions covered.")
    return "\n".join(lines)


def _save_coverage_png(imgpoints, size, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, h = size
    xs = np.concatenate([p.reshape(-1, 2)[:, 0] for p in imgpoints]) if imgpoints else np.array([])
    ys = np.concatenate([p.reshape(-1, 2)[:, 1] for p in imgpoints]) if imgpoints else np.array([])
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.scatter(xs, ys, s=4, alpha=0.4)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title("detected corner coverage (want edges + bottom corners filled)")
    for f in (1 / 3, 2 / 3):
        ax.axvline(w * f, color="0.8", lw=0.5); ax.axhline(h * f, color="0.8", lw=0.5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fisheye_calib_flags() -> int:
    """RECOMPUTE_EXTRINSIC + FIX_SKEW, version-proof.

    Some newer opencv-python wheels stopped exposing the
    `cv2.fisheye.CALIB_*` enum constants while keeping the functions
    (CI hit this on 2026-07-06 with tests green locally). The numeric
    values are ABI-stable in the C++ header (fisheye::CALIB_
    RECOMPUTE_EXTRINSIC = 1<<1, FIX_SKEW = 1<<3), so fall back to them.
    """
    # NB do NOT fall back to the top-level cv2.CALIB_* constants: in OpenCV 5
    # those are a DIFFERENT enum (RECOMPUTE_EXTRINSIC = 8388608, FIX_SKEW =
    # 33554432) belonging to calibrateCamera, not fisheye::. Substituting them
    # runs the calibration with neither flag actually set, and it still returns
    # a plausible-looking K, so the mistake is silent.
    recompute = int(getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", 1 << 1))
    fix_skew = int(getattr(cv2.fisheye, "CALIB_FIX_SKEW", 1 << 3))
    return recompute + fix_skew


def _check_opencv_fisheye_usable() -> None:
    """Refuse to run under OpenCV 5, whose fisheye.calibrate is broken.

    Measured 2026-08-19 on the same 24 correspondences: OpenCV 5.0.0 returns
    RMS 36.65 px and IGNORES the flags entirely (every flag value, including 0,
    gives byte-identical output, so CALIB_RECOMPUTE_EXTRINSIC never applies),
    while OpenCV 4.10 returns 2.03 px. It does not error: it returns a
    plausible-looking K, so the failure is silent and would be adopted.

    Only `fisheye.calibrate` is affected. `fisheye.undistortPoints` /
    `distortPoints` round-trip exactly under 5.0.0, so the rest of the pipeline
    is unaffected and does NOT need pinning.

    Run this tool under OpenCV 4:
        python3 -m venv /tmp/cvcal && /tmp/cvcal/bin/pip install \
            'opencv-python==4.10.0.84' 'numpy<2.3' pyyaml matplotlib
        PYTHONPATH=. /tmp/cvcal/bin/python -m scripts.sensor_processing.fisheye_recalibrate ...
    """
    # A stubbed solver (unit tests) is not the native one, so the version of
    # the underlying library is irrelevant: native OpenCV functions are
    # builtins, a Python stand-in is a plain function.
    if isinstance(cv2.fisheye.calibrate, types.FunctionType):
        return
    major = int(cv2.__version__.split(".")[0])
    if major >= 5:
        raise SystemExit(
            f"cv2.fisheye.calibrate is unreliable in OpenCV {cv2.__version__} "
            f"(returns ~36 px RMS and ignores flags; 4.10 gives ~2 px on the "
            f"same data). Re-run under OpenCV 4: see _check_opencv_fisheye_usable."
        )


def calibrate(images, cols, rows, square_mm):
    _check_opencv_fisheye_usable()
    objpoints, imgpoints, size, used, skipped = _find_corners(images, cols, rows)
    if len(objpoints) < 5:
        raise SystemExit(
            f"only {len(objpoints)} usable images (need >= 5; >=15 recommended). "
            f"Check the board size (--cols/--rows are INNER corners) and lighting."
        )
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = _fisheye_calib_flags()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, size, K, D, flags=flags, criteria=criteria)
    except cv2.error as exc:
        raise SystemExit(
            "cv2.fisheye.calibrate failed, usually the board views are too "
            "similar (all frontal/coplanar). Re-shoot with the board TILTED at "
            "varied out-of-plane angles (lean it toward/away and side to side), "
            "not just slid around the frame.\n"
            f"  OpenCV: {exc}"
        ) from exc

    # Per-image reprojection error (in pixels; square_mm only scales extrinsics).
    # fisheye.projectPoints and findChessboardCorners can return different array
    # shapes ((1,N,2) vs (N,1,2)), so flatten both to (N,2) before comparing.
    per_img = []
    for i in range(len(objpoints)):
        proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        a = np.asarray(imgpoints[i], dtype=np.float64).reshape(-1, 2)
        b = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
        per_img.append(err)
    return dict(K=K, D=D, size=size, rms=float(rms), per_img=per_img,
                used=used, skipped=skipped, imgpoints=imgpoints)


def _yaml_block(K, D, size) -> str:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).ravel()
    k = "\n".join(f"    - [{K[i,0]:.6f}, {K[i,1]:.6f}, {K[i,2]:.6f}]" for i in range(3))
    d = "\n".join(f"    - [{D[i]:.10f}]" for i in range(4))
    return (f"fisheye:\n  K:\n{k}\n  D:\n{d}\n"
            f"  image_size: [{size[0]}, {size[1]}]\n  model: fisheye")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="glob for calibration jpgs, e.g. 'data/fisheye_checkerboard/*.jpg'")
    ap.add_argument("--cols", type=int, default=10,
                    help="INNER corners across = (squares across - 1). Default 10 "
                         "to match docs/checkered_board_10x7.pdf, which has 10x7 "
                         "INNER corners (11x8 squares): confirmed by the corner "
                         "detector.")
    ap.add_argument("--rows", type=int, default=7,
                    help="INNER corners down = (squares down - 1). Default 7.")
    ap.add_argument("--square-mm", type=float, default=25.0,
                    help="square size in mm (only scales extrinsics; K/D unaffected)")
    ap.add_argument("--out", default=None, help="write calibration YAML here")
    ap.add_argument("--coverage-png", default=None, help="write a corner-coverage scatter PNG")
    args = ap.parse_args(argv)

    images = sorted(glob.glob(args.images))
    if not images:
        raise SystemExit(f"no images match {args.images!r}")
    print(f"found {len(images)} images; detecting {args.cols}x{args.rows} inner corners...")

    res = calibrate(images, args.cols, args.rows, args.square_mm)
    K, D, size = res["K"], res["D"], res["size"]
    print(f"\nused {len(res['used'])}/{len(images)} images "
          f"({len(res['skipped'])} skipped: board not found / unreadable)")
    print(f"overall reprojection error (RMS): {res['rms']:.3f} px "
          f"({'GOOD' if res['rms'] < 0.5 else 'HIGH: re-shoot or check board size'})")
    pe = np.array(res["per_img"])
    print(f"per-image error: min {pe.min():.2f}  median {np.median(pe):.2f}  max {pe.max():.2f} px")
    print()
    print(_coverage_report(res["imgpoints"], size))
    print(f"\nK=\n{np.round(K,2)}\nD={np.round(D.ravel(),5)}  image_size={size}")

    if args.coverage_png:
        _save_coverage_png(res["imgpoints"], size, args.coverage_png)
        print(f"\ncoverage scatter -> {args.coverage_png}")

    block = _yaml_block(K, D, size)
    if args.out:
        import datetime  # noqa: F401 - timestamp passed in by caller if needed
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            f"# Fisheye recalibration. RMS reprojection error = {res['rms']:.3f} px,\n"
            f"# {len(res['used'])} images, board {args.cols}x{args.rows}.\n"
            f"# Validate with range_validate.py before pasting into "
            f"configs/intrinsics.yaml.\n{block}\n")
        print(f"\ncalibration YAML -> {args.out}")
    print("\nTo adopt: validate with scripts.eval.range_validate, then paste the K/D/"
          "image_size above into the fisheye block of configs/intrinsics.yaml\n"
          "(keep its cx/pix_deg_ratio unless you also re-derive those).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
