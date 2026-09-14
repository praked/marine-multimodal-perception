"""sensor_health thermal discovery: by-id first, retried first read.

CLAUDE.md §27 (re-verified 2026-08-17): sensor_health's thermal probe was a
FALSE NEGATIVE twice on identical-working hardware — it walked /dev/video0-3
by index with a single un-retried read(), so the PureThermal's first-frame
latency read as "disconnected" while smoke_capture streamed 30/30. The probe
now mirrors continuous_capture._open_thermal: the /dev/v4l/by-id/*PureThermal*
symlink first with a retried first read, then the index walk with one cheap
probe per node so a dead node cannot stall the check. These tests pin that
contract with mocked device listing/capture (no hardware, no Pi).
"""
import glob

import numpy as np
import pytest

import cv2

from scripts.data_collection import sensor_health as sh


BY_ID_LINK = "/dev/v4l/by-id/usb-GroupGets_PureThermal_x-video-index0"
BY_ID_NODE = "/dev/video8"


def _thermal_frame():
    # Varying, mid-brightness frame so the health thresholds score PASS:
    # mean in (5, 250), spatial std > 1.5, temporal diff > 0.03.
    rng = np.random.default_rng()
    return rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)


@pytest.fixture
def quiet(monkeypatch):
    """Fresh report, no sleeping, deterministic realpath."""
    monkeypatch.setattr(sh, "REPORT", [])
    monkeypatch.setattr(sh.time, "sleep", lambda s: None)
    monkeypatch.setattr(sh.os.path, "realpath", lambda p: BY_ID_NODE)
    return monkeypatch


def _statuses():
    return {name: status for name, status, _ in sh.REPORT}


def test_by_id_node_is_retried_when_the_first_reads_are_empty(quiet):
    """The measured PureThermal failure: empty first reads on a healthy
    camera. The by-id node must be retried, not reported disconnected."""
    quiet.setattr(glob, "glob", lambda pat: [BY_ID_LINK] if "by-id" in pat else [])

    opened, reads = [], []

    class FlakyCap:
        def __init__(self, dev, *a):
            self.dev = dev
            opened.append(dev)

        def read(self):
            reads.append(self.dev)
            if len(reads) < 3:  # first two reads come back empty
                return False, None
            return True, _thermal_frame()

        def release(self):
            pass

    quiet.setattr(cv2, "VideoCapture", FlakyCap)
    frame = sh.test_thermal()
    assert frame is not None, "a healthy camera was rejected on its first read"
    assert _statuses()["thermal"] == "PASS"
    assert opened[0] == BY_ID_NODE, "by-id node must be probed before the index walk"
    assert reads[:3] == [BY_ID_NODE] * 3, "the empty first reads were not retried"


def test_scanned_nodes_get_a_single_probe_each(quiet):
    """No by-id symlink and nothing answers: the index-walk fallback keeps one
    cheap read per node (a dead node can block seconds per read) and the
    verdict is an honest FAIL."""
    quiet.setattr(glob, "glob", lambda pat: [])

    reads = []

    class DeadCap:
        def __init__(self, dev, *a):
            self.dev = dev

        def read(self):
            reads.append(self.dev)
            return False, None

        def release(self):
            pass

    quiet.setattr(cv2, "VideoCapture", DeadCap)
    assert sh.test_thermal() is None
    assert _statuses()["thermal"] == "FAIL"
    assert reads == [1, 0, 2, 3], "each scanned node gets exactly one probe"


def test_index_walk_still_finds_the_thermal_without_a_by_id_link(quiet):
    """Fallback path: no symlink, camera answers on an index, one read."""
    quiet.setattr(glob, "glob", lambda pat: [])

    class GoodCap:
        def __init__(self, dev, *a):
            self.dev = dev

        def read(self):
            return True, _thermal_frame()

        def release(self):
            pass

    quiet.setattr(cv2, "VideoCapture", GoodCap)
    assert sh.test_thermal() is not None
    assert _statuses()["thermal"] == "PASS"


def test_oversized_frames_are_rejected_as_non_thermal(quiet):
    """A node streaming fisheye-sized frames is not the Lepton: the walk must
    move past it and land on the real 160x120 node."""
    quiet.setattr(glob, "glob", lambda pat: [])

    class SizedCap:
        def __init__(self, dev, *a):
            self.dev = dev

        def read(self):
            if self.dev == 1:  # first index probed: a big RGB camera
                return True, np.zeros((648, 864, 3), dtype=np.uint8)
            if self.dev == 0:
                return True, _thermal_frame()
            return False, None

        def release(self):
            pass

    quiet.setattr(cv2, "VideoCapture", SizedCap)
    assert sh.test_thermal() is not None
    assert _statuses()["thermal"] == "PASS"
    detail = next(d for n, s, d in sh.REPORT if n == "thermal")
    assert detail.startswith("0 "), "the 160x120 node, not the big one, must win"


def test_module_import_runs_nothing():
    """The probes live behind main(): importing the module (as these tests do)
    must not touch hardware. A populated pristine REPORT at import time would
    mean the module-level driver came back. (The other tests monkeypatch
    REPORT, so the import-time list is untouched here.)"""
    assert sh.REPORT == []
