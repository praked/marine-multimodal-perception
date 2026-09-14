import sys
from unittest import mock

import pytest

import scripts.eval.healthcheck as hc


def test_check_ok(capsys):
    ok, detail = hc.check("a", lambda: "fine")
    assert ok is True
    assert detail == "fine"
    out = capsys.readouterr().out
    assert "OK" in out


def test_check_fail(capsys):
    ok, detail = hc.check("b", lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert ok is False
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_section(capsys):
    hc.section("title")
    out = capsys.readouterr().out
    assert "title" in out


def test_main_skip_tests_succeeds(capsys, monkeypatch, synthetic_triplet):
    """Drive main() against a controlled, known-good triplet so the exit
    code does not depend on the integrity of the local data/ dir."""
    monkeypatch.setattr(hc, "list_triplets", lambda: [synthetic_triplet])
    with mock.patch.object(sys, "argv", ["hc", "--skip-tests"]):
        with pytest.raises(SystemExit) as exc:
            hc.main()
        # SystemExit(0) = healthy
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "HEALTHY" in out or "summary" in out


def test_main_smoke_failure_branch(monkeypatch, capsys):
    """A triplet that raises during processing hits the per-triplet smoke
    FAIL branch (lines 86-88) and makes main exit nonzero."""
    class _FakeTriplet:
        clip_id = "Synth/boom"
        scene = "Synth"

    monkeypatch.setattr(hc, "list_triplets", lambda: [_FakeTriplet()])

    def _boom(*a, **k):
        raise RuntimeError("iterate exploded")

    # iterate_triplet is imported into the healthcheck namespace.
    monkeypatch.setattr(hc, "iterate_triplet", _boom)
    with mock.patch.object(sys, "argv", ["hc", "--skip-tests"]):
        with pytest.raises(SystemExit) as exc:
            hc.main()
        assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] Synth/boom: iterate exploded" in out


def test_main_pytest_success_branch(monkeypatch, capsys):
    """When pytest returns 0, the [OK] pytest line prints (line 106)."""
    class FakeResult:
        returncode = 0
        stdout = "5 passed in 0.1s\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    monkeypatch.setattr(hc, "list_triplets", lambda: [])
    with mock.patch.object(sys, "argv", ["hc"]):
        with pytest.raises(SystemExit) as exc:
            hc.main()
        # Empty triplets is not itself a failure; pytest passed.
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "[OK]   pytest" in out


def test_main_pytest_subprocess_raises(monkeypatch, capsys):
    """If subprocess.run itself raises (e.g. timeout), the except branch
    reports it and main exits nonzero (lines 111-113)."""
    def _raise(*a, **k):
        raise OSError("cannot spawn pytest")

    monkeypatch.setattr("subprocess.run", _raise)
    monkeypatch.setattr(hc, "list_triplets", lambda: [])
    with mock.patch.object(sys, "argv", ["hc"]):
        with pytest.raises(SystemExit) as exc:
            hc.main()
        assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "could not run pytest" in out


def test_main_with_failed_tests(monkeypatch, capsys):
    """If pytest is forced to return nonzero, healthcheck should exit nonzero."""
    class FakeResult:
        returncode = 1
        stdout = "FAILED something\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    # Keep the data smoke deterministic and fast: the failure under test is
    # the pytest run, not the data layer.
    monkeypatch.setattr(hc, "list_triplets", lambda: [])
    with mock.patch.object(sys, "argv", ["hc"]):
        with pytest.raises(SystemExit) as exc:
            hc.main()
        assert exc.value.code == 1
