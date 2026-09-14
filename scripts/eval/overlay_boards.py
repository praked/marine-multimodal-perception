"""Build overlay-comparison boards for the 36-frame smoke set.

Two products, both under one results subfolder:
  1. by_method/: one 6x6 contact sheet PER overlay method (all 36 frames with
     that overlay): none / segmentation / typed boxes / instance masks / all.
  2. by_frame/: one image per frame (36) showing that same frame under EVERY
     overlay side by side (including 'none').

Reuses the exact dashboard overlay routine (_overlay_camera), so the boards match
what the dashboard draws. Frames + overlays are resolved from the ORIGINAL
InstitutionOne frame_ids the smoke mapping points at (the eWaSR masks / YOLO dets /
YOLOv8-seg instances were all produced against those).

    python -m scripts.eval.overlay_boards \
        --mapping labels/eval_smoke36_mapping.json --out results/overlay_boards
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.eval.dashboard import CLASS_COLORS, _overlay_camera  # noqa: E402
from scripts.utils.detections import DetProvider, InstanceSegProvider  # noqa: E402
from scripts.utils.segmentation import SegProvider  # noqa: E402

VARIANTS = ["none", "groundingdino", "segmentation", "typed", "instance", "all"]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# GroundingDINO box class -> RGB, matching the dashboard's CLASS_COLORS exactly
# (boat=green, structure=orange, person=blue, buoy=red, duck=amber, ...).
GDINO_COLOURS = {k: _hex_to_rgb(v) for k, v in CLASS_COLORS.items()}


def _clipdir(fid: str) -> str:
    scene, clip_ts, _ = fid.split("/")
    return f"{scene}__{clip_ts}"


def _frame_jpg(frames_root: Path, fid: str) -> Path:
    scene, clip_ts, ts = fid.split("/")
    return frames_root / f"{scene}__{clip_ts}" / f"{ts}.jpg"


def _draw_gdino(rgb: np.ndarray, boxes: list) -> np.ndarray:
    """Draw the GroundingDINO bboxes (the prior standard detector, dashboard-
    audited, in labels/manual.jsonl) using the dashboard's per-class colours."""
    img = rgb.copy()
    h, w = img.shape[:2]
    for b in boxes or []:
        box = b.get("xyxy")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        if max(box) <= 1.5:
            x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
        cls = str(b.get("cls", "unset"))
        colour = GDINO_COLOURS.get(cls, GDINO_COLOURS["unset"])
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), colour, 2)
        cv2.putText(img, cls, (int(x0), max(10, int(y0) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return img


def render_variant(sr, variant, typed, inst, gdino) -> np.ndarray:
    kw = dict(show_horizon=False, show_projection=False, show_detections=False,
              show_seg=False)
    if variant == "groundingdino":
        return _draw_gdino(_overlay_camera(sr, **kw), gdino)
    if variant == "segmentation":
        kw["show_seg"] = True
    elif variant == "typed":
        kw["typed_dets"] = typed
    elif variant == "instance":
        kw["instance_dets"] = inst
    elif variant == "all":
        # all model outputs + the GroundingDINO boxes overlaid together
        kw.update(show_seg=True, typed_dets=typed, instance_dets=inst)
        return _draw_gdino(_overlay_camera(sr, **kw), gdino)
    return _overlay_camera(sr, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", default="labels/eval_smoke36_mapping.json")
    ap.add_argument("--frames-root", type=Path, default=Path("data/seg_frames"))
    ap.add_argument("--seg-root", default="data/seg")
    ap.add_argument("--det-root", default="data/det")
    ap.add_argument("--inst-root", default="data/det_seg")
    ap.add_argument("--out", type=Path, default=Path("results/overlay_boards"))
    args = ap.parse_args(argv)

    import json
    mapping = json.load(open(args.mapping))
    mp = mapping["fake_to_original"]
    orig_to_fake = mapping.get("original_to_fake", {v: k for k, v in mp.items()})
    # Stable order = the eval-clip (fake) timestamp order, matching the labelling board.
    frames = [mp[k] for k in sorted(mp)]

    # GroundingDINO boxes (prior standard detector, dashboard-audited), in
    # labels/manual.jsonl keyed by the eval-clip id.
    gdino_by_fake: dict[str, list] = {}
    mpath = Path("labels/manual.jsonl")
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fid = r.get("frame_id", "")
            if "eval_smoke36" in fid:
                gdino_by_fake[fid] = r.get("fisheye_bboxes", [])

    seg = SegProvider(args.seg_root)
    det = DetProvider(args.det_root)
    inst = InstanceSegProvider(args.inst_root)

    (args.out / "by_frame").mkdir(parents=True, exist_ok=True)

    # Pre-render every (frame, variant) once; reuse for both board types.
    rendered: list[dict] = []
    index_rows = []
    for i, fid in enumerate(frames):
        jpg = _frame_jpg(args.frames_root, fid)
        if not jpg.exists():
            print(f"  [skip] missing frame jpg: {jpg}")
            continue
        und = cv2.imread(str(jpg))  # BGR (what _overlay_camera expects)
        sr = SimpleNamespace(undistorted=und, horizon_line=(0.0, 0.0, 0.0),
                             coords=[], sizes=[], seg_mask=seg.get(fid))
        typed = det.get(fid) or []
        insts = inst.get(fid) or []
        gdino = gdino_by_fake.get(orig_to_fake.get(fid, ""), [])
        imgs = {v: render_variant(sr, v, typed, insts, gdino) for v in VARIANTS}
        rendered.append({"fid": fid, "imgs": imgs})
        index_rows.append({"idx": i, "frame_id": fid, "n_gdino": len(gdino),
                           "n_typed": len(typed), "n_instances": len(insts)})

        # by_frame panel: all variants for this frame
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        axes = axes.ravel()
        for ax, v in zip(axes, VARIANTS):
            ax.imshow(imgs[v])
            ax.set_title(v, fontsize=10)
            ax.axis("off")
        axes[-1].axis("off")
        scene, clip_ts, ts = fid.split("/")
        fig.suptitle(f"frame {i:02d}   {clip_ts}  {ts}", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out / "by_frame" / f"frame_{i:02d}.png", dpi=85)
        plt.close(fig)

    n = len(rendered)
    cols = 6
    rows = (n + cols - 1) // cols
    for v in VARIANTS:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.3))
        axes = np.array(axes).ravel()
        for j, r in enumerate(rendered):
            axes[j].imshow(r["imgs"][v])
            axes[j].set_title(f"{j:02d}", fontsize=7)
            axes[j].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"smoke-36: overlay: {v}", fontsize=14)
        fig.tight_layout()
        fig.savefig(args.out / f"board_{v}.png", dpi=110)
        plt.close(fig)

    with open(args.out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "frame_id", "n_gdino", "n_typed",
                                          "n_instances"])
        w.writeheader()
        w.writerows(index_rows)

    print(f"Done. {n} frames.")
    print(f"  by-method boards: {args.out}/board_<{'|'.join(VARIANTS)}>.png")
    print(f"  per-frame panels: {args.out}/by_frame/frame_NN.png")
    print(f"  index: {args.out}/index.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
