"""Box fingerprint: collection degrades, and the diff classifies correctly.

The classification is the load-bearing part. A Pi 4 -> Pi 5 swap legitimately
changes the model, serial, kernel and UART overlay; a second box legitimately
changes hostname and host keys. If those land in the same bucket as a changed
calibration file, the real regression gets waved through.
"""
from __future__ import annotations

import json

import pytest

from scripts.data_collection import box_fingerprint as bf


# --- classification --------------------------------------------------------

@pytest.mark.parametrize("key", sorted(bf.BOARD_KEYS))
def test_board_keys_classify_as_board(key):
    assert bf.classify(key) == "board"


@pytest.mark.parametrize("key", sorted(bf.IDENTITY_KEYS))
def test_identity_keys_classify_as_identity(key):
    assert bf.classify(key) == "identity"


@pytest.mark.parametrize("key", [
    "repocfg.extrinsics.yaml",     # the calibration itself
    "repocfg.imu.yaml",
    "cfg.box_env",                 # per-box constants (thermal rotation)
    "repo.commit",
    "sw.apt_manifest",
    "models.ewasr",
    "auth.sudo_nopasswd",
    "dev.ttyAMA0",
])
def test_everything_else_is_substance(key):
    assert bf.classify(key) == "substance"


def test_board_and_identity_sets_do_not_overlap():
    assert not (bf.BOARD_KEYS & bf.IDENTITY_KEYS)


# --- diff ------------------------------------------------------------------

def test_compare_sorts_differences_into_the_three_buckets():
    a = {"hw.model": "Pi 4", "id.hostname": "box-a",
         "repocfg.imu.yaml": "aaa", "sw.python": "3.13.0"}
    b = {"hw.model": "Pi 5", "id.hostname": "box-b",
         "repocfg.imu.yaml": "bbb", "sw.python": "3.13.0"}
    d = bf.compare(a, b)
    assert [k for k, _, _ in d["board"]] == ["hw.model"]
    assert [k for k, _, _ in d["identity"]] == ["id.hostname"]
    assert [k for k, _, _ in d["substance"]] == ["repocfg.imu.yaml"]


def test_identical_fingerprints_differ_nowhere():
    a = {"hw.model": "Pi 4", "repocfg.imu.yaml": "aaa"}
    assert bf.compare(a, dict(a)) == {"board": [], "identity": [],
                                      "substance": []}


def test_a_board_swap_alone_is_not_a_regression():
    """The whole point: swapping the board must come out clean on substance."""
    pi4 = {"hw.model": "Raspberry Pi 4 Model B", "hw.serial": "aaa",
           "os.kernel": "6.18.34+rpt-rpi-v8", "os.kernel_flavour": "pi4",
           "boot.uart_lines": "core_freq=500; core_freq_min=500; dtoverlay=miniuart-bt; enable_uart=1",
           "repocfg.extrinsics.yaml": "cal1", "repo.commit": "abc123"}
    pi5 = dict(pi4, **{"hw.model": "Raspberry Pi 5 Model B", "hw.serial": "bbb",
                       "os.kernel": "6.18.34+rpt-rpi-2712",
                       "os.kernel_flavour": "pi5",
                       "boot.uart_lines": "dtparam=uart0=on"})
    d = bf.compare(pi4, pi5)
    assert d["substance"] == []
    assert len(d["board"]) == 5


def test_a_changed_calibration_is_a_regression_even_across_a_swap():
    pi4 = {"hw.model": "Pi 4", "repocfg.extrinsics.yaml": "cal1"}
    pi5 = {"hw.model": "Pi 5", "repocfg.extrinsics.yaml": "cal2"}
    d = bf.compare(pi4, pi5)
    assert [k for k, _, _ in d["substance"]] == ["repocfg.extrinsics.yaml"]


def test_a_missing_field_is_reported_rather_than_ignored():
    """A field that vanishes (a config deleted on the new card) must surface."""
    d = bf.compare({"cfg.box_env": "abc"}, {})
    assert d["substance"] == [("cfg.box_env", "abc", "<missing>")]


# --- collection ------------------------------------------------------------

def test_collect_returns_scalars_only():
    """The diff prints one line per field, so nested values would break it."""
    for key, value in bf.collect().items():
        assert not isinstance(value, (dict, list, tuple)), key


def test_collect_survives_a_machine_that_is_not_a_pi():
    fp = bf.collect()
    assert "hw.model" in fp and "repo.commit" in fp
    # Absent hardware reports None/False rather than raising.
    assert fp["dev.rtc0"] in (True, False)


def test_helpers_do_not_raise_on_absent_paths_or_tools():
    assert bf._read("/definitely/not/here") is None
    assert bf._file_digest("/definitely/not/here") is None
    assert bf._run("definitely-not-a-real-binary-xyz") is None
    assert bf._ok("definitely-not-a-real-binary-xyz") is False
    assert bf._sha256(None) is None


def test_digest_is_stable_and_short():
    assert bf._sha256("hello") == bf._sha256("hello")
    assert len(bf._sha256("hello")) == 16


# --- cli -------------------------------------------------------------------

def test_compare_cli_passes_when_only_board_fields_differ(tmp_path, capsys):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps({"hw.model": "Pi 4", "repo.commit": "abc",
                             "_note": "before"}))
    b.write_text(json.dumps({"hw.model": "Pi 5", "repo.commit": "abc",
                             "_note": "after"}))
    assert bf.main(["--compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "before" in out and "after" in out


def test_compare_cli_fails_on_a_substantive_difference(tmp_path, capsys):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps({"repocfg.imu.yaml": "aaa"}))
    b.write_text(json.dumps({"repocfg.imu.yaml": "bbb"}))
    assert bf.main(["--compare", str(a), str(b)]) == 1
    assert "FAIL: 1 substantive difference" in capsys.readouterr().out


def test_out_writes_valid_json_with_the_note(tmp_path):
    p = tmp_path / "fp.json"
    assert bf.main(["-o", str(p), "--note", "Pi4 before imaging"]) == 0
    assert json.loads(p.read_text())["_note"] == "Pi4 before imaging"


def test_rtc_driver_is_substance_not_board():
    """A Pi 5 can let its built-in RTC claim rtc0, pushing the battery-backed
    DS3231 to rtc1. That breaks offline time while dev.rtc0 still reads True,
    so the driver name must be treated as a regression, not a board quirk."""
    assert bf.classify("dev.rtc0_driver") == "substance"
    d = bf.compare({"dev.rtc0": True, "dev.rtc0_driver": "rtc-ds1307"},
                   {"dev.rtc0": True, "dev.rtc0_driver": "rp1_rtc"})
    assert [k for k, _, _ in d["substance"]] == ["dev.rtc0_driver"]


def test_scripts_digest_pins_the_deployed_code(tmp_path, monkeypatch):
    """The box is an rsync copy, not a git clone, so repo.commit is null there.
    Without a content digest a hand-edited script on the Pi would sail through
    the 'nothing changed' check."""
    root = tmp_path / "scripts"
    (root / "sub").mkdir(parents=True)
    (root / "a.py").write_text("print(1)\n")
    (root / "sub" / "b.py").write_text("print(2)\n")
    monkeypatch.setattr(bf, "REPO", tmp_path)

    first = bf._scripts_digest()
    assert first["repo.scripts_files"] == 2
    assert bf._scripts_digest() == first          # stable

    (root / "sub" / "b.py").write_text("print(3)\n")
    assert bf._scripts_digest()["repo.scripts_digest"] != first["repo.scripts_digest"]


def test_scripts_digest_ignores_pycache(tmp_path, monkeypatch):
    root = tmp_path / "scripts"
    (root / "__pycache__").mkdir(parents=True)
    (root / "a.py").write_text("x=1\n")
    (root / "__pycache__" / "a.py").write_text("compiled junk\n")
    monkeypatch.setattr(bf, "REPO", tmp_path)
    assert bf._scripts_digest()["repo.scripts_files"] == 1


def test_scripts_digest_is_substance():
    assert bf.classify("repo.scripts_digest") == "substance"


# --- SSH auth posture ------------------------------------------------------
# This field silently reported "default(no)" on a box that had just had
# password auth enabled (2026-08-21): the drop-ins are mode 600 so an
# unprivileged read saw nothing, and the parser took the LAST match when sshd
# takes the FIRST. A confident wrong answer is worse than no answer here.

def test_password_auth_prefers_sshd_itself(monkeypatch):
    monkeypatch.setattr(bf, "sshd_effective", lambda k: "yes")
    assert bf._password_auth() == "yes"


def test_password_auth_reports_unknown_when_files_are_unreadable(monkeypatch):
    """The exact 2026-08-21 situation: drop-ins are mode 600, so the read
    returns nothing and the old code concluded "default(no)" on a box that had
    password auth switched ON."""
    monkeypatch.setattr(bf, "sshd_effective", lambda k: None)
    monkeypatch.setattr(bf, "_read", lambda p: None)          # permission denied
    monkeypatch.setattr(bf.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(bf.Path, "glob",
                        lambda self, pat: iter([bf.Path("50-cloud-init.conf")]))
    assert bf._password_auth() == "unknown (unreadable)"


def test_password_auth_honours_first_match_not_last(monkeypatch):
    """sshd takes the first value it obtains for a keyword."""
    monkeypatch.setattr(bf, "sshd_effective", lambda k: None)
    monkeypatch.setattr(bf, "_read",
                        lambda p: "PasswordAuthentication no\n"
                                  "PasswordAuthentication yes\n")
    monkeypatch.setattr(bf.Path, "is_dir", lambda self: False)
    assert bf._password_auth() == "no"


def test_password_auth_ignores_commented_lines(monkeypatch):
    monkeypatch.setattr(bf, "sshd_effective", lambda k: None)
    monkeypatch.setattr(bf, "_read", lambda p: "#PasswordAuthentication yes\n")
    monkeypatch.setattr(bf.Path, "is_dir", lambda self: False)
    assert bf._password_auth() == "default(no)"


def test_sshd_effective_parses_the_T_output(monkeypatch):
    monkeypatch.setattr(bf, "_run", lambda *c, **k:
                        "usepam yes\npasswordauthentication yes\nport 22")
    assert bf.sshd_effective("PasswordAuthentication") == "yes"
    assert bf.sshd_effective("NotAKeyword") is None
