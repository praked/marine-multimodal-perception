import json
import sys
from unittest import mock

import pytest

from scripts.data.release import (
    _count_labels,
    _hash_file,
    build_manifest,
    main,
    release,
)


def test_count_labels_missing_file(tmp_path):
    out = _count_labels(tmp_path / "absent.jsonl")
    assert out == {"total": 0, "audited": 0, "by_scene": {}}


def test_count_labels_aggregates(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"scene": "Boats", "audited": True}) + "\n" +
        json.dumps({"scene": "Boats", "audited": False}) + "\n" +
        json.dumps({"scene": "Ducks", "audited": True}) + "\n"
    )
    out = _count_labels(p)
    assert out["total"] == 3
    assert out["audited"] == 2
    assert out["by_scene"] == {"Boats": 2, "Ducks": 1}


def test_count_labels_skips_blank(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n\n" + json.dumps({"scene": "X", "audited": True}) + "\n")
    assert _count_labels(p)["total"] == 1


def test_hash_file_present(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello")
    assert len(_hash_file(p)) == 16


def test_hash_file_absent(tmp_path):
    assert _hash_file(tmp_path / "nope") is None


def test_build_manifest_has_fields():
    m = build_manifest("test")
    assert m["version"] == "test"
    assert "splits" in m
    assert "labels" in m
    assert "config_hashes" in m


def test_release_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("scripts.data.release.RELEASES_DIR", tmp_path)
    out = release("dry", dry_run=True)
    assert out is None
    assert not (tmp_path / "dry").exists()
    msg = capsys.readouterr().out
    assert "dry-run" in msg


def test_release_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.data.release.RELEASES_DIR", tmp_path)
    out = release("v1", dry_run=False)
    assert out is not None and out.exists()
    assert (out / "manifest.json").exists()
    m = json.loads((out / "manifest.json").read_text())
    assert m["version"] == "v1"


def test_release_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.data.release.RELEASES_DIR", tmp_path)
    (tmp_path / "v1").mkdir()
    with pytest.raises(SystemExit):
        release("v1", dry_run=False)


def test_main_dry_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("scripts.data.release.RELEASES_DIR", tmp_path)
    with mock.patch.object(sys, "argv", ["release", "--version", "test",
                                          "--dry-run"]):
        main()
    out = capsys.readouterr().out
    assert "test" in out
