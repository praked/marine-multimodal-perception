"""Set curation: which clips are deleted and which frame ranges are cut.

Source of truth is the dashboard's `sail_curation` table (soft delete +
restore + trim ranges per set, edited in the web UI). `pnpm curation:export`
in `dashboard/` writes it to **`configs/curation.yaml`** (git-tracked), and
this module reads that file so every offline consumer — `list_triplets`,
`iterate_triplet`, the bundle baker, the feature exporter, the audit planner —
sees the same curation without a database connection.

Schema (`docs/reference/data_formats.md` §6)::

    version: 1
    exported_at: "2026-08-28T20:00:00Z"
    sets:
      <scene>__<first_chunk_ts>:          # dashboard clip key
        deleted_at: "…" | null            # deleted := deleted_at set and not
        restored_at: "…" | null           #   restored since, or purged
        purged_at: "…" | null
        chunks: ["<ts>", …]               # member capture chunks (an activity
                                          #   concatenates successive chunks)
        cuts:
          - {start_ts: "HH:MM:SS.f", end_ts: "HH:MM:SS.f", note: "…"}
        note: "…"

Keys resolve two ways: a clip id (`scene/ts` or `scene__ts`) matches a set
directly, or through the set's `chunks` list — so deleting a concatenated
activity in the dashboard deletes every capture chunk it was built from,
and a cut on the activity applies to whichever chunk holds those frames
(cuts are wall-clock ranges, chunk-independent by construction).

Frame ids are never rewritten by curation: labels, audits and sector
records keyed by frame id keep lining up; consumers simply skip the frames.

Nothing here touches files on disk: curation is metadata. The raw captures
on the SSD are never deleted by anything that reads this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "configs" / "curation.yaml"
#: Env override (tests, alternative exports). Empty/unset = the default path.
ENV_PATH = "ASVPROJECT_CURATION"


def ts_seconds(ts: str) -> float:
    """'HH:MM:SS.f' (or safe_ts 'HH-MM-SS.f') -> seconds of day."""
    h, m, s = str(ts).replace("-", ":").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalise_clip_key(clip_id: str) -> str:
    """'scene/ts' or 'scene__ts' -> 'scene__ts' (the dashboard key)."""
    return str(clip_id).replace("/", "__")


@dataclass(frozen=True)
class Cut:
    start_ts: str
    end_ts: str
    note: str = ""

    def contains(self, ts: str) -> bool:
        a, b = ts_seconds(self.start_ts), ts_seconds(self.end_ts)
        lo, hi = min(a, b), max(a, b)
        return lo <= ts_seconds(ts) <= hi


def in_cut(cuts, ts: str) -> bool:
    """True when `ts` lies inside any cut (inclusive both ends)."""
    return any(c.contains(ts) for c in cuts)


@dataclass(frozen=True)
class CurationSet:
    key: str                     # 'scene__first_ts'
    scene: str
    chunks: tuple[str, ...]      # member chunk timestamps
    deleted_at: str | None
    restored_at: str | None
    purged_at: str | None
    cuts: tuple[Cut, ...]
    note: str = ""

    @property
    def deleted(self) -> bool:
        if self.purged_at:
            return True
        if not self.deleted_at:
            return False
        return not self.restored_at or self.restored_at < self.deleted_at


class Curation:
    """The parsed curation file. Cheap to query; build once via load()."""

    def __init__(self, sets: dict[str, CurationSet] | None = None,
                 source: Path | None = None):
        self.sets: dict[str, CurationSet] = dict(sets or {})
        self.source = source
        # chunk key -> set (activity membership)
        self._by_chunk: dict[str, CurationSet] = {}
        for s in self.sets.values():
            for ts in s.chunks:
                self._by_chunk.setdefault(f"{s.scene}__{ts}", s)

    # ---- construction ---------------------------------------------------
    @classmethod
    def empty(cls) -> "Curation":
        return cls({}, None)

    @classmethod
    def from_dict(cls, data: dict, source: Path | None = None) -> "Curation":
        sets: dict[str, CurationSet] = {}
        for key, raw in (data.get("sets") or {}).items():
            raw = raw or {}
            key = normalise_clip_key(key)
            scene, _, first_ts = key.partition("__")
            chunks = tuple(str(c) for c in (raw.get("chunks") or [first_ts]))
            cuts = tuple(
                Cut(str(c["start_ts"]), str(c["end_ts"]), str(c.get("note") or ""))
                for c in (raw.get("cuts") or [])
                if c and c.get("start_ts") is not None and c.get("end_ts") is not None
            )
            sets[key] = CurationSet(
                key=key, scene=scene, chunks=chunks,
                deleted_at=_opt_str(raw.get("deleted_at")),
                restored_at=_opt_str(raw.get("restored_at")),
                purged_at=_opt_str(raw.get("purged_at")),
                cuts=cuts, note=str(raw.get("note") or ""))
        return cls(sets, source)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Curation":
        """Read the YAML; a missing file is an empty curation (nothing
        deleted, nothing cut) so pre-curation checkouts behave as before."""
        p = Path(path) if path is not None else resolve_path()
        if not p.exists():
            return cls.empty()
        import yaml
        with open(p) as fh:
            data = yaml.safe_load(fh) or {}
        if int(data.get("version", 1)) != 1:
            raise ValueError(f"{p}: unsupported curation version {data.get('version')}")
        return cls.from_dict(data, p)

    # ---- queries --------------------------------------------------------
    def set_for(self, clip_id: str) -> CurationSet | None:
        key = normalise_clip_key(clip_id)
        return self.sets.get(key) or self._by_chunk.get(key)

    def is_deleted(self, clip_id: str) -> bool:
        s = self.set_for(clip_id)
        return bool(s and s.deleted)

    def cuts_for(self, clip_id: str) -> tuple[Cut, ...]:
        s = self.set_for(clip_id)
        return s.cuts if s else ()

    def keep_frame(self, clip_id: str, ts: str) -> bool:
        """False for a frame of a deleted set or inside one of its cuts."""
        s = self.set_for(clip_id)
        if s is None:
            return True
        if s.deleted:
            return False
        return not in_cut(s.cuts, ts)

    def deleted_keys(self) -> set[str]:
        return {k for k, s in self.sets.items() if s.deleted}

    def __len__(self) -> int:
        return len(self.sets)


def _opt_str(v) -> str | None:
    return None if v in (None, "") else str(v)


def resolve_path() -> Path:
    env = os.environ.get(ENV_PATH, "").strip()
    return Path(env) if env else DEFAULT_PATH


_cache: dict[str, tuple[float, Curation]] = {}


def load_curation(path: str | Path | None = None) -> Curation:
    """Cached `Curation.load` keyed by path + mtime (re-reads after an
    export). The hot loops (iterate_triplet per clip) call this freely."""
    p = Path(path) if path is not None else resolve_path()
    key = str(p)
    mtime = p.stat().st_mtime if p.exists() else -1.0
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    cur = Curation.load(p)
    _cache[key] = (mtime, cur)
    return cur


def is_deleted(clip_id: str) -> bool:
    return load_curation().is_deleted(clip_id)


def keep_frame(clip_id: str, ts: str) -> bool:
    return load_curation().keep_frame(clip_id, ts)


def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect configs/curation.yaml")
    ap.add_argument("--path", default=None)
    args = ap.parse_args(argv)
    cur = load_curation(args.path)
    print(f"{cur.source or '(none)'}: {len(cur)} set(s), "
          f"{len(cur.deleted_keys())} deleted")
    for key, s in sorted(cur.sets.items()):
        flag = "DELETED" if s.deleted else f"{len(s.cuts)} cut(s)"
        print(f"  {key}: {flag}" + (f"  # {s.note}" if s.note else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
