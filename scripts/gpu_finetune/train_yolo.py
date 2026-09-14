"""Fine-tune a YOLO detector on the exported ASVProject dataset (GPU node).

Run after sourcing env.sh (so caches/outputs land on scratch). All paths should
point at scratch; pretrained weights download to the CWD, which env.sh sets to
scratch.

    python train_yolo.py \
        --data /scratch0/$USER/yolo_finetune/data.yaml \
        --project /scratch0/$USER/yolo_runs --name smoke --epochs 100

The dataset is produced locally by scripts.eval.export_yolo and rsynced up.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to data.yaml on scratch")
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained base (nano = Pi-realistic)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=864, help="864 matches the undistorted frame width")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", required=True, help="output dir on scratch")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    ap.add_argument("--patience", type=int, default=50)
    args = ap.parse_args(argv)

    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=args.project, name=args.name, device=args.device,
        patience=args.patience, exist_ok=True, plots=True, verbose=True,
    )
    outdir = Path(args.project) / args.name
    print(f"\ndone. best weights: {outdir/'weights'/'best.pt'}")
    print(f"results + curves under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
