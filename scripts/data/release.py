"""Snapshot a versioned dataset release: Branch B.1.

Bundles the current labels, configs, split definition, and a
clip manifest into a single tagged directory under
`labels/releases/<version>/`. Reproduces a dataset state for later
comparisons; also produces a one-page manifest readable by humans.

Usage:
    python -m scripts.data.release --version 2026.06
    python -m scripts.data.release --version dryrun --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from scripts.data.splits import SplitConfig, assign_split
from scripts.utils.datasets import REPO_ROOT, list_triplets

LABELS_DIR = REPO_ROOT / "labels"
RELEASES_DIR = LABELS_DIR / "releases"
CONFIGS_DIR = REPO_ROOT / "configs"


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _count_labels(path: Path) -> dict:
    if not path.exists():
        return {"total": 0, "audited": 0, "by_scene": {}}
    total = 0
    audited = 0
    by_scene: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            if obj.get("audited"):
                audited += 1
            scene = obj.get("scene", "?")
            by_scene[scene] = by_scene.get(scene, 0) + 1
    return {"total": total, "audited": audited, "by_scene": by_scene}


def build_manifest(version: str) -> dict:
    cfg = SplitConfig.load()
    triplets = list_triplets()
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for t in triplets:
        splits[assign_split(t.clip_id, cfg)].append(t.clip_id)

    label_files = {
        "manual": LABELS_DIR / "manual.jsonl",
        "qwen":   LABELS_DIR / "qwen.jsonl",
        "master": LABELS_DIR / "master.jsonl",
    }
    label_stats = {k: _count_labels(p) for k, p in label_files.items()}

    config_hashes = {
        "intrinsics.yaml":  _hash_file(CONFIGS_DIR / "intrinsics.yaml"),
        "detection.yaml":   _hash_file(CONFIGS_DIR / "detection.yaml"),
        "extrinsics.yaml":  _hash_file(CONFIGS_DIR / "extrinsics.yaml"),
        "splits.yaml":      _hash_file(CONFIGS_DIR / "splits.yaml"),
    }

    return {
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_triplets": len(triplets),
        "splits": {k: sorted(v) for k, v in splits.items()},
        "labels": label_stats,
        "config_hashes": config_hashes,
    }


def release(version: str, dry_run: bool = False) -> Path | None:
    out_dir = RELEASES_DIR / version
    manifest = build_manifest(version)
    print(f"=== dataset release v{version} ===")
    print(f"triplets : {manifest['n_triplets']}")
    print(f"splits   : train={len(manifest['splits']['train'])}  "
          f"val={len(manifest['splits']['val'])}  "
          f"test={len(manifest['splits']['test'])}")
    for k, s in manifest["labels"].items():
        print(f"labels {k:6s}: total={s['total']}  audited={s['audited']}  by_scene={s['by_scene']}")

    if dry_run:
        print("(dry-run: nothing written)")
        return None

    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing release {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for f in ("manual.jsonl", "qwen.jsonl", "master.jsonl"):
        src = LABELS_DIR / f
        if src.exists():
            shutil.copy2(src, out_dir / f)
    for f in ("intrinsics.yaml", "detection.yaml", "extrinsics.yaml", "splits.yaml"):
        src = CONFIGS_DIR / f
        if src.exists():
            shutil.copy2(src, out_dir / f"config_{f}")
    print(f"wrote -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True,
                    help="Release tag, e.g. 2026.06 or pre_institutionone_baseline.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the manifest without writing.")
    args = ap.parse_args()
    release(args.version, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
