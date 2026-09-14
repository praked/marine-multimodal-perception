import pytest
import yaml

from scripts.data.splits import SplitConfig, _clip_date, _hash_bucket, assign_split


def test_clip_date_parses():
    d = _clip_date("Boats/2025-06-23_16-21-07")
    assert d is not None
    assert d.year == 2025 and d.month == 6 and d.day == 23


def test_clip_date_invalid_returns_none():
    assert _clip_date("garbage") is None
    assert _clip_date("Scene/notadate") is None


def test_hash_bucket_deterministic():
    a = _hash_bucket("X/Y")
    b = _hash_bucket("X/Y")
    assert a == b
    assert 0.0 <= a < 1.0


def test_hash_bucket_varies():
    a = _hash_bucket("X/1")
    b = _hash_bucket("X/2")
    assert a != b


def test_assign_split_pinned_val():
    cfg = SplitConfig(test_after=None,
                      val_clips={"Boats/abc"},
                      test_clips=set(),
                      ratios={"train": 0.7, "val": 0.15, "test": 0.15})
    assert assign_split("Boats/abc", cfg) == "val"


def test_assign_split_pinned_test():
    cfg = SplitConfig(test_after=None,
                      val_clips=set(),
                      test_clips={"Boats/abc"},
                      ratios={"train": 0.7, "val": 0.15, "test": 0.15})
    assert assign_split("Boats/abc", cfg) == "test"


def test_assign_split_test_after():
    import datetime as dt
    cfg = SplitConfig(test_after=dt.date(2026, 6, 1),
                      val_clips=set(),
                      test_clips=set(),
                      ratios={"train": 0.7, "val": 0.15, "test": 0.15})
    assert assign_split("X/2026-06-15_00-00-00", cfg) == "test"
    # Pre-cutoff -> hash bucket.
    s = assign_split("X/2024-01-01_00-00-00", cfg)
    assert s in ("train", "val", "test")


def test_assign_split_uses_default_when_missing():
    s = assign_split("Boats/2025-06-23_16-21-07")
    assert s in ("train", "val", "test")


def test_split_config_load_missing_file_uses_defaults(tmp_path):
    cfg = SplitConfig.load(tmp_path / "absent.yaml")
    assert cfg.test_after is None
    assert cfg.ratios["train"] == 0.7


def test_split_config_load_real(repo_root):
    cfg = SplitConfig.load(repo_root / "configs" / "splits.yaml")
    assert "Boats/2025-06-23_16-21-07" in cfg.val_clips


def test_main_lists_splits(capsys):
    import sys
    from unittest import mock
    from scripts.data.splits import main
    with mock.patch.object(sys, "argv", ["splits", "--split", "all"]):
        main()
    out = capsys.readouterr().out
    assert "train" in out and "val" in out and "test" in out


@pytest.mark.needs_data
def test_main_single_split(capsys):
    import sys
    from unittest import mock
    from scripts.data.splits import main
    with mock.patch.object(sys, "argv", ["splits", "--split", "val"]):
        main()
    out = capsys.readouterr().out
    assert "Boats/2025-06-23_16-21-07" in out


def test_split_config_load_from_temp(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({
        "test_after": "2026-01-01",
        "val_clips": ["X/a"],
        "test_clips": ["X/b"],
        "pre_test_after_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
    }))
    cfg = SplitConfig.load(p)
    import datetime as dt
    assert cfg.test_after == dt.date(2026, 1, 1)
    assert "X/a" in cfg.val_clips
    assert "X/b" in cfg.test_clips
    assert cfg.ratios["train"] == 0.5


def test_group_hash_same_outing_same_split(tmp_path):
    """Under group_hash, clips from one scene+day always co-assign."""
    import yaml
    from scripts.data.splits import SplitConfig, assign_split

    cfg_p = tmp_path / "splits.yaml"
    cfg_p.write_text(yaml.safe_dump({
        "test_after": None, "val_clips": [], "test_clips": [],
        "pre_test_after_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
        "group_hash": True, "groups": {},
    }))
    cfg = SplitConfig.load(cfg_p)
    a = assign_split("2026-07-15/2026-07-15_09-00-00", cfg)
    b = assign_split("2026-07-15/2026-07-15_16-45-12", cfg)
    assert a == b


def test_group_hash_explicit_group_overrides(tmp_path):
    """An explicit groups: entry wins over the scene/date key."""
    import yaml
    from scripts.data.splits import SplitConfig, group_key

    cfg_p = tmp_path / "splits.yaml"
    cfg_p.write_text(yaml.safe_dump({
        "group_hash": True,
        "groups": {"harbour-north": ["2026-07-15/2026-07-15_09-00-00"]},
    }))
    cfg = SplitConfig.load(cfg_p)
    assert group_key("2026-07-15/2026-07-15_09-00-00", cfg) == "harbour-north"
    assert group_key("2026-07-15/2026-07-15_16-45-12", cfg) == \
        "2026-07-15/2026-07-15"
    # Flat clip_id (no scene prefix) must not produce a leading slash.
    assert group_key("2026-07-15_16-45-12", cfg) == "2026-07-15"


def test_group_hash_off_matches_legacy(tmp_path):
    """group_hash: false (and absent) keeps the per-clip hashing."""
    import yaml
    from scripts.data.splits import SplitConfig, _hash_bucket, assign_split

    cfg_p = tmp_path / "splits.yaml"
    cfg_p.write_text(yaml.safe_dump({
        "pre_test_after_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    }))
    cfg = SplitConfig.load(cfg_p)
    assert cfg.group_hash is False
    clip = "Boats/2025-06-23_16-21-07"
    b = _hash_bucket(clip)
    expected = "train" if b < 0.7 else ("val" if b < 0.85 else "test")
    assert assign_split(clip, cfg) == expected
