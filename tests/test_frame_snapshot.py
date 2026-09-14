"""frame_snapshot: the latest-frame channel between capture and shadow."""
import json

import numpy as np

from scripts.data_collection.frame_snapshot import (
    FISHEYE_NAME, META_NAME, THERMAL_NAME, SnapshotReader, SnapshotWriter,
)


def _frame(v, shape=(6, 8, 3)):
    return np.full(shape, v, dtype=np.uint8)


def test_roundtrip_and_key(tmp_path):
    w = SnapshotWriter(tmp_path / "snap")
    r = SnapshotReader(tmp_path / "snap")
    assert r.read() is None                      # nothing published yet
    assert w.publish("2099-01-01_00-00-00", 3, "00:00:01.0", _frame(7), _frame(9, (4, 5, 3)))
    snap = r.read()
    assert snap is not None and snap.seq == 1
    assert snap.chunk_ts == "2099-01-01_00-00-00" and snap.frame_index == 3
    assert snap.fisheye.shape == (6, 8, 3) and int(snap.fisheye[0, 0, 0]) == 7
    assert snap.thermal is not None and int(snap.thermal[0, 0, 0]) == 9
    # same seq is not re-delivered unless asked
    assert r.read() is None
    assert r.read(allow_repeat=True).seq == 1
    # no tmp files linger; meta written last
    names = {p.name for p in (tmp_path / "snap").iterdir()}
    assert names == {FISHEYE_NAME, THERMAL_NAME, META_NAME}


def test_want_key_refuses_other_frames(tmp_path):
    w = SnapshotWriter(tmp_path)
    r = SnapshotReader(tmp_path)
    w.publish("C", 10, "00:00:03.3", _frame(1))
    assert r.read(want=("C", 9)) is None         # a slow reader never gets N+1's pixels for row N
    assert r.read(want=("D", 10)) is None        # different chunk
    snap = r.read(want=("C", 10))
    assert snap is not None and snap.thermal is None and snap.frame_index == 10


def test_writer_never_raises(tmp_path):
    w = SnapshotWriter(tmp_path)
    assert w.publish("C", 0, "t", None) is False  # nothing to publish
    # an unwritable directory is counted, not raised
    w.directory = tmp_path / "missing" / "deeper"
    assert w.publish("C", 1, "t", _frame(1)) is False
    assert w.errors == 1 and w.last_error


def test_meta_is_json_with_index(tmp_path):
    w = SnapshotWriter(tmp_path)
    w.publish("2099-01-01_00-00-00", 42, "00:00:14.0", _frame(3))
    meta = json.loads((tmp_path / META_NAME).read_text())
    assert meta["frame_index"] == 42 and meta["has_thermal"] is False and meta["seq"] == 1


def test_want_tolerance_accepts_one_frame_newer_only(tmp_path):
    w = SnapshotWriter(tmp_path)
    r = SnapshotReader(tmp_path)
    w.publish("C", 11, "00:00:03.6", _frame(2))
    assert r.read(want=("C", 10), allow_repeat=True) is None                 # exact by default
    snap = r.read(want=("C", 10), allow_repeat=True, tolerance=1)            # one frame newer: ok
    assert snap is not None and snap.frame_index == 11
    assert r.read(want=("C", 9), allow_repeat=True, tolerance=1) is None     # two frames newer: no
    assert r.read(want=("C", 12), allow_repeat=True, tolerance=1) is None    # older than the row: never
    assert r.read(want=("D", 11), allow_repeat=True, tolerance=1) is None    # other chunk: never
