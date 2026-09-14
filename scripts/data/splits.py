"""Frozen train / val / test split conventions: Branch B.1.

Defines deterministic assignment of clip IDs to data splits, driven by
`configs/splits.yaml`. The contract:

  - `test_after` (date string YYYY-MM-DD): every triplet whose timestamp
    is on or after this date is `test`. Captures from before are split
    by scene-stratified hashing into `train` / `val` / `test` in
    `pre_test_after_ratios`.
  - `val_clips` and `test_clips` (lists of clip IDs): explicit overrides.
    Take precedence over the date rule.
  - `group_hash` (bool, default false): hash on the clip's GROUP key
    instead of the clip_id, so clips from the same outing land in the
    same split: same-outing clips are near-duplicates and per-clip
    hashing leaks them across train/val (approved 2026-07-10,
    docs/history/2026-07-09_thermal_tuning.md splits proposal). Automatic group
    key = `<scene>/<YYYY-MM-DD>` (same scene, same day = same outing);
    the optional `groups:` map ({group_name: [clip_id, ...]}) overrides
    it for days spanning multiple locations (one-word location tag at
    capture: runbook §3.4).

The split for a given clip_id is *stable*: re-running the function
returns the same answer for the same `(clip_id, splits.yaml)` pair.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from scripts.utils.datasets import REPO_ROOT, list_triplets

SPLITS_PATH = REPO_ROOT / "configs" / "splits.yaml"

DEFAULT_CONFIG = {
    # Once InstitutionOne field captures land, freeze the test set by the
    # arrival date. Until then, all existing data is split by hash.
    "test_after": None,
    "val_clips": [],
    "test_clips": [],
    "pre_test_after_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    "group_hash": False,
    "groups": {},
}


@dataclass
class SplitConfig:
    test_after: date | None
    val_clips: set[str]
    test_clips: set[str]
    ratios: dict[str, float]
    group_hash: bool = False
    # clip_id -> group name (inverted from the yaml's {group: [clips]}).
    groups: dict[str, str] | None = None

    @classmethod
    def load(cls, path: str | Path = SPLITS_PATH) -> "SplitConfig":
        if not Path(path).exists():
            raw = dict(DEFAULT_CONFIG)
        else:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            for k, v in DEFAULT_CONFIG.items():
                raw.setdefault(k, v)
        ta = raw.get("test_after")
        ta_date = datetime.strptime(ta, "%Y-%m-%d").date() if ta else None
        clip_to_group = {c: g for g, clips in (raw.get("groups") or {}).items()
                         for c in (clips or [])}
        return cls(
            test_after=ta_date,
            val_clips=set(raw.get("val_clips") or []),
            test_clips=set(raw.get("test_clips") or []),
            ratios=dict(raw.get("pre_test_after_ratios") or DEFAULT_CONFIG["pre_test_after_ratios"]),
            group_hash=bool(raw.get("group_hash", False)),
            groups=clip_to_group,
        )


def _clip_date(clip_id: str) -> date | None:
    """Extract YYYY-MM-DD from `<scene>/<ts>` where ts starts with the date."""
    try:
        ts = clip_id.split("/")[-1]
        return datetime.strptime(ts[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _hash_bucket(clip_id: str) -> float:
    """Return a float in [0,1) deterministic in clip_id."""
    h = int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16)
    return (h % 10000) / 10000.0


def group_key(clip_id: str, cfg: SplitConfig) -> str:
    """The unit that gets hashed under `group_hash`: the outing, not the
    clip: explicit `groups:` entry if present, else `<scene>/<YYYY-MM-DD>`
    (same scene, same day = same outing = near-duplicate content)."""
    if cfg.groups and clip_id in cfg.groups:
        return cfg.groups[clip_id]
    scene, _, ts = clip_id.rpartition("/")
    # A flat clip_id with no scene prefix (rpartition -> scene == "") must not
    # gain a leading slash: "/2026-07-08" would never match a real group key.
    return f"{scene}/{ts[:10]}" if scene else ts[:10]


def assign_split(clip_id: str, cfg: SplitConfig | None = None) -> str:
    cfg = cfg or SplitConfig.load()
    if clip_id in cfg.test_clips:
        return "test"
    if clip_id in cfg.val_clips:
        return "val"
    d = _clip_date(clip_id)
    if cfg.test_after and d and d >= cfg.test_after:
        return "test"
    # Hash-based pre-cutoff split (per-clip, or per-outing under group_hash).
    b = _hash_bucket(group_key(clip_id, cfg) if cfg.group_hash else clip_id)
    train_thresh = cfg.ratios["train"]
    val_thresh = train_thresh + cfg.ratios["val"]
    if b < train_thresh:
        return "train"
    if b < val_thresh:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    ap.add_argument("--config", default=str(SPLITS_PATH))
    args = ap.parse_args()

    cfg = SplitConfig.load(args.config)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for t in list_triplets():
        out[assign_split(t.clip_id, cfg)].append(t.clip_id)

    if args.split == "all":
        for k in ("train", "val", "test"):
            print(f"## {k} ({len(out[k])})")
            for c in sorted(out[k]):
                print(f"  {c}")
    else:
        for c in sorted(out[args.split]):
            print(c)


if __name__ == "__main__":
    main()
