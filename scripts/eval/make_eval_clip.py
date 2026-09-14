"""Assemble the smoke-set frames into a single dashboard-loadable triplet.

Builds a "fake" clip (fisheye mp4 + thermal mp4 + empty mmwave csv) under
``data/captures/<mission>/`` from the frames listed in a frames-file (default:
the detector smoke set). Annotate this clip once in the dashboard to get a
ground-truth label set, then score any detector run against it.

Crucial detail: the dashboard UNDISTORTS the fisheye on load, so the mp4 must
hold the **raw** (distorted) frames, exactly what the original capture mp4s
hold, not the undistorted ones. We therefore pull the raw fisheye + thermal
frames straight out of ``iterate_triplet`` (pre-undistortion).

Because the fake clip re-times the frames, its ``frame_id``s differ from the
originals. A mapping (fake_frame_id <-> original_frame_id) is written so a
detector run on the ORIGINAL frames can be remapped and scored against the
manual labels drawn on the fake clip (or vice versa), no re-run needed.

    python -m scripts.eval.make_eval_clip \\
        --captures-dir data/captures/2026-06-17_institutionone_day1 \\
        --frames-file labels/qwen/smoke_set.txt \\
        --out-mission eval_smoke36 --ts 2026-06-17_18-00-00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.datasets import CAPTURES_DIR, resolve_triplet


def _format_hhmmss(t_sec: float) -> str:
    t_sec %= 86400
    h, rem = divmod(t_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures-dir", required=True,
                    help="Source mission folder holding the original clips.")
    ap.add_argument("--frames-file", required=True,
                    help="frame_ids to include (one per line), in order.")
    ap.add_argument("--out-mission", default="eval_smoke36",
                    help="New mission folder name under data/captures/.")
    ap.add_argument("--ts", default="2026-06-17_18-00-00",
                    help="Clip timestamp suffix (YYYY-MM-DD_HH-MM-SS).")
    ap.add_argument("--fps", type=float, default=3.0)
    args = ap.parse_args()

    src_dir = Path(args.captures_dir).expanduser().resolve()
    order = [ln.split("#", 1)[0].strip()
             for ln in Path(args.frames_file).read_text().splitlines()
             if ln.split("#", 1)[0].strip()]
    wanted = set(order)

    from scripts.utils.calibration import load_detection
    detection = load_detection()

    # Collect raw fisheye + thermal frames for each wanted frame_id.
    raw: dict[str, tuple] = {}  # frame_id -> (fisheye_raw, thermal_raw)
    clips = sorted({fid.split("/")[1] for fid in wanted if "/" in fid})
    for clip_ts in clips:
        triplet = resolve_triplet(src_dir / clip_ts)
        for ts, fish_raw, therm_raw, _pts in iterate_triplet(triplet, detection):
            fid = f"{triplet.scene}/{triplet.timestamp}/ts={ts.replace(':', '-')}"
            if fid in wanted:
                raw[fid] = (fish_raw.copy(), therm_raw.copy())
    missing = [f for f in order if f not in raw]
    if missing:
        print(f"!! {len(missing)} frame_ids not found (skipped): {missing[:3]}...")
    order = [f for f in order if f in raw]
    if not order:
        raise SystemExit("no frames resolved")

    out_dir = CAPTURES_DIR / args.out_mission
    out_dir.mkdir(parents=True, exist_ok=True)
    fish_path = out_dir / f"fisheye_{args.ts}.mp4"
    therm_path = out_dir / f"thermal_{args.ts}.mp4"
    mmwave_path = out_dir / f"mmwave_{args.ts}.csv"

    fh, fw = raw[order[0]][0].shape[:2]
    th, tw = raw[order[0]][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fish_w = cv2.VideoWriter(str(fish_path), fourcc, args.fps, (fw, fh))
    therm_w = cv2.VideoWriter(str(therm_path), fourcc, args.fps, (tw, th))

    mapping = {}  # fake_frame_id -> original_frame_id (and reverse)
    base = 18 * 3600  # synthetic clip start; arbitrary but stable
    for i, fid in enumerate(order):
        fish, therm = raw[fid]
        fish_w.write(fish)
        therm_w.write(cv2.resize(therm, (tw, th)))
        fake_ts = _format_hhmmss(base + i / args.fps).replace(":", "-")
        fake_fid = f"{args.out_mission}/{args.ts}/ts={fake_ts}"
        mapping[fake_fid] = fid
    fish_w.release()
    therm_w.release()
    # Empty mmwave -> dashboard uses the fps-synthesised timestamp path.
    mmwave_path.write_text("Date,Time,X,Y,Z\n")

    map_path = Path("labels") / f"{args.out_mission}_mapping.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps({
        "fake_to_original": mapping,
        "original_to_fake": {v: k for k, v in mapping.items()},
        "source_captures_dir": str(src_dir),
        "frames_file": args.frames_file,
    }, indent=2))

    print(f"wrote {len(order)} frames -> {fish_path}")
    print(f"  thermal: {therm_path}  ({tw}x{th})")
    print(f"  mmwave : {mmwave_path} (empty)")
    print(f"  mapping: {map_path}")
    print(f"\nOpen in the dashboard:")
    print(f"  python -m scripts.eval.dashboard --triplet "
          f"data/captures/{args.out_mission}/{args.ts}")


if __name__ == "__main__":
    main()
