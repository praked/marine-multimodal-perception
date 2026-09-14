"""Run model.val() and print PER-CLASS mAP (runs ON THE GPU NODE).

Works for both a detection model and a YOLOv8-seg model (prints Box metrics, and
Mask metrics too when present). The point is to see which of the 8 LaRS classes
(boat/buoy/swimmer/...) are actually trustworthy vs dragged-to-zero by the
training class imbalance: the aggregate mAP hides this.

    python yolo_val_perclass.py --model .../best.pt --data .../data.yaml --imgsz 960
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def _per_class(metric, names, idx):
    """Yield (name, mAP50, mAP50-95) for the classes present in this metric."""
    for i, c in enumerate(idx):
        yield names[int(c)], float(metric.ap50[i]), float(metric.ap[i])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args(argv)

    m = YOLO(args.model)
    names = m.names
    metrics = m.val(data=args.data, imgsz=args.imgsz, verbose=True)

    print("\n================ PER-CLASS (Box) ================")
    box = metrics.box
    print(f"{'class':14s} {'mAP50':>8s} {'mAP50-95':>9s}")
    for name, ap50, ap in _per_class(box, names, box.ap_class_index):
        print(f"{name:14s} {ap50:8.3f} {ap:9.3f}")
    print(f"{'ALL':14s} {box.map50:8.3f} {box.map:9.3f}")

    seg = getattr(metrics, "seg", None)
    if seg is not None:
        print("\n================ PER-CLASS (Mask) ===============")
        print(f"{'class':14s} {'mAP50':>8s} {'mAP50-95':>9s}")
        for name, ap50, ap in _per_class(seg, names, seg.ap_class_index):
            print(f"{name:14s} {ap50:8.3f} {ap:9.3f}")
        print(f"{'ALL':14s} {seg.map50:8.3f} {seg.map:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
