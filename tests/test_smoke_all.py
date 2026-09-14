import sys
from unittest import mock

import pytest

from scripts.eval.smoke_all import main, run_one


def test_run_one_synthetic(intrinsics_real, detection_real, synthetic_triplet):
    r = run_one(synthetic_triplet, intrinsics_real, detection_real)
    assert r["n_frames"] == 5
    assert 0.0 <= r["mean_score"] <= 1.0
    assert 0.0 <= r["mean_max"] <= 1.0
    assert 0.0 <= r["hit_rate"] <= 1.0


@pytest.mark.needs_data
def test_main_runs_with_scene_filter(capsys):
    with mock.patch.object(sys, "argv", ["smoke", "--scene", "Boats"]):
        main()
    out = capsys.readouterr().out
    assert "Boats/" in out
    assert "all" in out and "OK" in out


def test_main_no_triplets_exits(monkeypatch):
    monkeypatch.setattr("scripts.eval.smoke_all.list_triplets", lambda s=None: [])
    with mock.patch.object(sys, "argv", ["smoke"]):
        with pytest.raises(SystemExit):
            main()


class _FakeTriplet:
    clip_id = "Synth/fake"
    scene = "Synth"


def test_main_clip_failure_non_strict(monkeypatch, capsys):
    """A clip whose processing raises is reported as FAIL and main exits
    nonzero (lines 75-77, 83-84) when not strict."""
    monkeypatch.setattr("scripts.eval.smoke_all.list_triplets",
                        lambda s=None: [_FakeTriplet()])
    monkeypatch.setattr("scripts.eval.smoke_all.load_intrinsics", lambda: {})
    monkeypatch.setattr("scripts.eval.smoke_all.load_detection",
                        lambda: {"fusion": {}})

    def _boom(*a, **k):
        raise RuntimeError("clip exploded")

    monkeypatch.setattr("scripts.eval.smoke_all.run_one", _boom)
    with mock.patch.object(sys, "argv", ["smoke"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "FAIL: clip exploded" in out
    assert "1 clips failed" in out


def test_main_clip_failure_strict_exits_immediately(monkeypatch, capsys):
    """With --strict, the first failure prints a traceback and exits 2
    (lines 78-80)."""
    monkeypatch.setattr("scripts.eval.smoke_all.list_triplets",
                        lambda s=None: [_FakeTriplet()])
    monkeypatch.setattr("scripts.eval.smoke_all.load_intrinsics", lambda: {})
    monkeypatch.setattr("scripts.eval.smoke_all.load_detection",
                        lambda: {"fusion": {}})

    def _boom(*a, **k):
        raise RuntimeError("strict explode")

    monkeypatch.setattr("scripts.eval.smoke_all.run_one", _boom)
    with mock.patch.object(sys, "argv", ["smoke", "--strict"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "FAIL: strict explode" in out
