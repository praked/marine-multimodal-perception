"""scripts.data.repair_videos: moov recovery + thermal stuck-row repair copies.

Videos are synthesised with OpenCV's own mp4v writer (the capture writer), so
the healthy/broken pair matches what the box produces: a moov-less file is
made by truncating a healthy one before its moov atom.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.data import repair_videos as rv

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _write(path: Path, n: int, w=160, h=120, hot_row=None, seed=0):
    rng = np.random.default_rng(seed)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
    for _ in range(n):
        f = rng.integers(40, 200, size=(h, w, 3)).astype(np.uint8)
        if hot_row is not None:
            f[hot_row] = 246
        wr.write(f)
    wr.release()


def _truncate_before_moov(path: Path) -> Path:
    buf = path.read_bytes()
    r = rv.find_box(buf, ["moov"])
    assert r is not None
    start = r[0] - 8  # box header precedes the payload
    broken = path.with_name(path.name.replace("-18.mp4", "-19.mp4"))
    assert broken != path
    broken.write_bytes(buf[:start])
    return broken


def test_box_parser_finds_moov_and_esds(tmp_path):
    p = tmp_path / "thermal_2026-08-26_10-00-18.mp4"
    _write(p, 12)
    assert rv.has_moov(p)
    hdr = rv.vol_header(p)
    assert hdr.startswith(b"\x00\x00\x01\xb0") or hdr.startswith(b"\x00\x00\x01\x20")
    assert rv.mdat_payload_offset(p) is not None


def test_recover_moov_from_truncated_file(tmp_path):
    healthy = tmp_path / "fisheye_2026-08-26_10-00-18.mp4"
    _write(healthy, 15)
    broken = _truncate_before_moov(healthy)
    assert not rv.has_moov(broken)
    assert rv.frame_count(broken) == 0
    out = tmp_path / "rec.mp4"
    n = rv.recover_moov(broken, healthy, out)
    assert n == 15
    assert rv.frame_count(out) == 15


def test_hot_row_detected_and_repaired_copy(tmp_path):
    p = tmp_path / "thermal_2026-08-26_10-00-18.mp4"
    _write(p, 20, hot_row=98)
    row, excess = rv.hot_row(p)
    assert row == 98 and excess > 60
    out = tmp_path / "rec.mp4"
    n, ok = rv.repair_rows_video(p, out, 98)
    assert ok and n == 20
    cap = cv2.VideoCapture(str(out))
    _, f = cap.read()
    cap.release()
    g = f.mean(axis=2)
    assert abs(g[98].mean() - (g[97].mean() + g[99].mean()) / 2) < 8  # codec noise only


def test_clean_video_has_no_hot_row(tmp_path):
    p = tmp_path / "thermal_2026-08-26_10-00-18.mp4"
    _write(p, 20)
    assert rv.hot_row(p)[0] is None


def test_repair_mission_end_to_end(tmp_path):
    root = tmp_path / "2026-08-26_afloat"
    root.mkdir()
    _write(root / "fisheye_2026-08-26_10-00-18.mp4", 10)
    broken = _truncate_before_moov(root / "fisheye_2026-08-26_10-00-18.mp4")
    (root / "mmwave_2026-08-26_10-00-19.csv").write_text("Date,Time,X,Y,Z\n2026-08-26,10:00:19.1,0,1,0\n")
    _write(root / "thermal_2026-08-26_10-00-18.mp4", 10, hot_row=98)
    (root / "fisheye_2026-08-26_10-00-20.mp4").write_bytes(b"")
    log = rv.repair_mission(root)
    joined = "\n".join(log)
    assert "MOOV     fisheye_2026-08-26_10-00-19.mp4 -> 10 frames" in joined
    assert "ROW98" in joined and "OK" in joined
    assert "EMPTY    fisheye_2026-08-26_10-00-20.mp4" in joined
    assert (root / "recovered" / broken.name).exists()
    assert (root / "recovered" / "thermal_2026-08-26_10-00-18.mp4").exists()
    # originals untouched, second run skips
    assert not rv.has_moov(broken)
    assert all(l.startswith(("SKIP", "EMPTY", "CLEAN")) for l in rv.repair_mission(root))
    # dry run on a fresh copy writes nothing
    root2 = tmp_path / "dry"
    shutil.copytree(root, root2, ignore=shutil.ignore_patterns("recovered"))
    assert any(l.startswith("MOOV?") for l in rv.repair_mission(root2, dry_run=True))
    assert not (root2 / "recovered").exists()
