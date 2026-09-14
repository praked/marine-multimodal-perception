"""list_triplets must see per-boot session sub-folders (captures/<mission>/<stamp>/)."""
from pathlib import Path

import scripts.utils.datasets as ds


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")


def test_list_triplets_descends_into_session_subfolders(tmp_path, monkeypatch):
    cap = tmp_path / "captures"
    m = cap / "2026-09-08_afloat"
    for ts in ("2026-09-08_17-11-13", "2026-09-08_17-16-13"):
        for stream in ("fisheye", "thermal"):
            _touch(m / "2026-09-08_17-10-55" / f"{stream}_{ts}.mp4")
        _touch(m / "2026-09-08_17-10-55" / f"mmwave_{ts}.csv")
    _touch(m / "_shadow" / "shadow_x.jsonl")                       # underscore dirs are skipped
    for stream in ("fisheye", "thermal"):                             # a flat chunk still counts
        _touch(m / f"{stream}_2026-09-08_09-00-00.mp4")
    _touch(m / "mmwave_2026-09-08_09-00-00.csv")
    monkeypatch.setattr(ds, "CAPTURES_DIR", cap)
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path / "nodata")
    monkeypatch.setattr(ds, "list_capture_missions", lambda: ["2026-09-08_afloat"])
    got = sorted(t.clip_id for t in ds.list_triplets(respect_curation=False))
    assert got == ["2026-09-08_afloat/2026-09-08_09-00-00",
                   "2026-09-08_afloat_2026-09-08_17-10-55/2026-09-08_17-11-13",
                   "2026-09-08_afloat_2026-09-08_17-10-55/2026-09-08_17-16-13"], got
