"""Set curation (configs/curation.yaml) — loader semantics + the consumers
that must honour it: list_triplets, iterate_triplet, the dashboard baker,
build_features and the audit planner. Everything runs on SYNTHETIC
fixtures in tmp_path: the SSD/data footage is never required (and never
touched), so nothing here skips silently when the SSD is unmounted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils import curation as cu
from scripts.utils import datasets as ds
from scripts.utils.curation import Curation, Cut, in_cut, load_curation
from scripts.utils.datasets import Triplet, list_triplets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dashboard" / "tools"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENE = "SynthCur"
TS = "2099-07-01_12-00-00"
N_FRAMES = 8


def _make_triplet(base: Path, scene: str = SCENE, ts: str = TS,
                  n: int = N_FRAMES) -> Triplet:
    """Synthetic complete triplet: n frames at 3 fps, one radar group per
    frame (V+SNR+NOISE columns), timestamps 12:00:00.0 … step 1 s."""
    scene_dir = base / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    fish = scene_dir / f"fisheye_{ts}.mp4"
    therm = scene_dir / f"thermal_{ts}.mp4"
    mm = scene_dir / f"mmwave_{ts}.csv"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(therm), fourcc, 3.0, (160, 120))
    for i in range(n):
        f = np.full((120, 160, 3), 60 + i * 5, dtype=np.uint8)
        f[40:80, 60:100] = 200
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()
    rows = ["Date,Time,X,Y,Z,V,SNR,NOISE"]
    for i in range(n):
        rows.append(f"2099-07-01,12:00:{i:02d}.0,0.1,1.5,0.0,0.2,14.5,8.0")
    mm.write_text("\n".join(rows) + "\n")
    return Triplet(scene=scene, timestamp=ts, fisheye=fish, thermal=therm,
                   mmwave=mm)


def _write_curation(path: Path, sets: dict) -> Path:
    path.write_text(yaml.safe_dump(
        {"version": 1, "exported_at": "2026-08-28T20:00:00Z", "sets": sets}))
    return path


@pytest.fixture
def curation_file(tmp_path, monkeypatch):
    """A curation.yaml with one deleted set (by activity key + chunks) and
    one trimmed set, activated via the ASVPROJECT_CURATION env override."""
    p = _write_curation(tmp_path / "curation.yaml", {
        f"{SCENE}__2099-07-01_11-00-00": {
            "deleted_at": "2026-08-28T10:00:00Z",
            "chunks": ["2099-07-01_11-00-00", "2099-07-01_11-05-00"],
            "cuts": [], "note": "indoor",
        },
        f"{SCENE}__{TS}": {
            "deleted_at": None, "restored_at": None, "purged_at": None,
            "chunks": [TS],
            "cuts": [{"start_ts": "12:00:02.0", "end_ts": "12:00:04.0",
                      "note": "operator in frame"},
                     {"start_ts": "12:00:07.0", "end_ts": "12:00:07.0"}],
        },
        "Other__2099-01-01_00-00-00": {
            "deleted_at": "2026-08-01T00:00:00Z",
            "restored_at": "2026-08-02T00:00:00Z",   # restored: active
            "chunks": ["2099-01-01_00-00-00"], "cuts": [],
        },
    })
    monkeypatch.setenv(cu.ENV_PATH, str(p))
    cu._cache.clear()
    return p


# ---------------------------------------------------------------------------
# Loader semantics
# ---------------------------------------------------------------------------

def test_missing_file_is_empty_curation(tmp_path, monkeypatch):
    monkeypatch.setenv(cu.ENV_PATH, str(tmp_path / "absent.yaml"))
    cu._cache.clear()
    cur = load_curation()
    assert len(cur) == 0
    assert cur.is_deleted("Any/2026-01-01_00-00-00") is False
    assert cur.keep_frame("Any/2026-01-01_00-00-00", "00:00:00.0") is True


def test_deleted_by_key_and_by_chunk_membership(curation_file):
    cur = load_curation()
    # activity key, chunk clip_id, chunk key — all resolve to the deleted set
    assert cur.is_deleted(f"{SCENE}__2099-07-01_11-00-00")
    assert cur.is_deleted(f"{SCENE}/2099-07-01_11-00-00")
    assert cur.is_deleted(f"{SCENE}/2099-07-01_11-05-00")
    assert cur.is_deleted(f"{SCENE}__2099-07-01_11-05-00")
    assert not cur.is_deleted(f"{SCENE}/{TS}")
    # a restore after the deletion makes the set active again
    assert not cur.is_deleted("Other/2099-01-01_00-00-00")
    assert cur.deleted_keys() == {f"{SCENE}__2099-07-01_11-00-00"}


def test_purged_is_deleted_even_when_restored_earlier():
    cur = Curation.from_dict({"sets": {"s__t": {
        "deleted_at": "2026-07-01T00:00:00Z",
        "restored_at": "2026-07-02T00:00:00Z",
        "purged_at": "2026-08-01T00:00:00Z"}}})
    assert cur.is_deleted("s/t")


def test_cuts_inclusive_both_ends_and_frame_keeping(curation_file):
    cur = load_curation()
    cid = f"{SCENE}/{TS}"
    assert [c.start_ts for c in cur.cuts_for(cid)] == ["12:00:02.0", "12:00:07.0"]
    assert cur.keep_frame(cid, "12:00:01.9")
    assert not cur.keep_frame(cid, "12:00:02.0")
    assert not cur.keep_frame(cid, "12:00:03.5")
    assert not cur.keep_frame(cid, "12:00:04.0")
    assert cur.keep_frame(cid, "12:00:04.1")
    assert not cur.keep_frame(cid, "12:00:07.0")
    # safe_ts dashes accepted; deleted set keeps nothing
    assert not cur.keep_frame(cid, "12-00-03.0")
    assert not cur.keep_frame(f"{SCENE}/2099-07-01_11-05-00", "11:06:00.0")
    # reversed ends normalise
    assert in_cut((Cut("12:00:05.0", "12:00:03.0"),), "12:00:04.0")


def test_load_curation_cache_follows_mtime(tmp_path, monkeypatch):
    p = _write_curation(tmp_path / "c.yaml", {})
    monkeypatch.setenv(cu.ENV_PATH, str(p))
    cu._cache.clear()
    assert len(load_curation()) == 0
    _write_curation(p, {"a__b": {"deleted_at": "2026-08-28T00:00:00Z"}})
    import os
    os.utime(p, (p.stat().st_atime + 5, p.stat().st_mtime + 5))
    assert load_curation().is_deleted("a/b")


def test_unsupported_version_raises(tmp_path):
    p = tmp_path / "v9.yaml"
    p.write_text("version: 9\nsets: {}\n")
    with pytest.raises(ValueError):
        Curation.load(p)


def test_cli_lists_sets(curation_file, capsys):
    assert cu._cli([]) == 0
    out = capsys.readouterr().out
    assert "3 set(s), 1 deleted" in out
    assert "2 cut(s)" in out


# ---------------------------------------------------------------------------
# list_triplets / iterate_triplet
# ---------------------------------------------------------------------------

def test_list_triplets_skips_deleted_sets(tmp_path, monkeypatch, curation_file):
    _make_triplet(tmp_path / "data", ts=TS)
    _make_triplet(tmp_path / "data", ts="2099-07-01_11-05-00")  # deleted chunk
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ds, "CAPTURES_DIR", tmp_path / "data" / "captures")
    monkeypatch.setattr(ds, "SCENES", (SCENE,))
    kept = [t.timestamp for t in list_triplets()]
    assert kept == [TS]
    everything = [t.timestamp for t in list_triplets(respect_curation=False)]
    assert everything == ["2099-07-01_11-05-00", TS]


def test_iterate_triplet_skips_cut_frames_in_lockstep(tmp_path, curation_file,
                                                      detection_real):
    trip = _make_triplet(tmp_path / "data")
    kept = [ts for ts, _, _, _ in iterate_triplet(trip, detection_real, {})]
    assert kept == ["12:00:00.0", "12:00:01.0", "12:00:05.0", "12:00:06.0"]
    raw = list(iterate_triplet(trip, detection_real, {}, respect_curation=False))
    assert [r[0] for r in raw] == [f"12:00:0{i}.0" for i in range(N_FRAMES)]
    # the frame AFTER a cut is the right video frame, not a stale one: it
    # decodes identically to the same timestamp of the raw walk, and the
    # synthetic frames brighten by 5 per index so it differs from the frame
    # the cut started on.
    by_ts = {ts: f for ts, f, _, _ in iterate_triplet(trip, detection_real, {})}
    raw_by_ts = {r[0]: r[1] for r in raw}
    assert np.array_equal(by_ts["12:00:05.0"], raw_by_ts["12:00:05.0"])
    assert abs(float(by_ts["12:00:05.0"].mean())
               - float(raw_by_ts["12:00:02.0"].mean())) > 8


def test_iterate_triplet_byte_identical_without_curation(tmp_path, monkeypatch,
                                                         detection_real):
    monkeypatch.setenv(cu.ENV_PATH, str(tmp_path / "absent.yaml"))
    cu._cache.clear()
    trip = _make_triplet(tmp_path / "data")
    a = list(iterate_triplet(trip, detection_real, {}))
    b = list(iterate_triplet(trip, detection_real, {}, respect_curation=False))
    assert [x[0] for x in a] == [x[0] for x in b]
    assert len(a) == N_FRAMES


def test_iterate_triplet_fallback_timeline_also_skips_cuts(tmp_path, curation_file,
                                                           detection_real):
    trip = _make_triplet(tmp_path / "data")
    trip.mmwave.write_text("Date,Time,X,Y,Z\n")   # empty radar -> synthesized ts
    kept = [ts for ts, _, _, _ in iterate_triplet(trip, detection_real, {})]
    # synthesized 12:00:00.0 + i/3 s; cut 12:00:02.0-04.0 drops i=6..12
    assert "12:00:01.7" in kept
    assert all(not (2.0 <= float(t.split(":")[-1]) <= 4.0) for t in kept)


# ---------------------------------------------------------------------------
# Baker (dashboard/tools/bake_corpus.py)
# ---------------------------------------------------------------------------

def test_baker_select_chunks_drops_deleted_before_grouping(tmp_path, curation_file):
    import bake_corpus as bc
    cur = load_curation()
    trips = [
        Triplet(SCENE, "2099-07-01_11-00-00", Path("a"), None, None),   # deleted key
        Triplet(SCENE, "2099-07-01_11-05-00", Path("a"), None, None),   # deleted via chunks
        Triplet(SCENE, TS, Path("a"), None, None),                       # trimmed, kept
        Triplet("eval_smoke36", TS, Path("a"), None, None),              # policy scene
        Triplet("Boats", "2025-06-23_16-21-07", Path("a"), None, None),  # pre-campaign
    ]
    kept = bc.select_chunks(trips, cur)
    assert [(t.scene, t.timestamp) for t in kept] == [(SCENE, TS)]
    # --ignore-curation keeps the deleted chunks (policy scenes still out)
    kept_all = bc.select_chunks(trips, Curation.empty())
    assert len(kept_all) == 3


def test_baker_does_not_bake_cut_frames(tmp_path, curation_file, intrinsics_real,
                                        detection_real):
    import bake_corpus as bc
    trip = _make_triplet(tmp_path / "data")
    clip_dir = tmp_path / "bundle" / f"{SCENE}__{TS}"
    frames_index: list[dict] = []
    sectors: dict = {}
    radar: dict = {}
    K, D = intrinsics_real["fisheye"]["K"], intrinsics_real["fisheye"]["D"]
    bc._bake_chunk_into(trip, TS, clip_dir, 1, K, D, detection_real,
                        frames_index, sectors, radar, {}, {}, {})
    baked = sorted(f["ts"] for f in frames_index)
    assert baked == ["12:00:00.0", "12:00:01.0", "12:00:05.0", "12:00:06.0"]
    on_disk = sorted(p.name for p in (clip_dir / "frames").glob("*.jpg"))
    assert "ts=12-00-03.0.jpg" not in on_disk
    assert "ts=12-00-05.0.jpg" in on_disk
    assert set(radar) == set(baked)   # cut frames contribute no radar group


def test_baker_every_sampling_anchored_to_raw_index(tmp_path, curation_file,
                                                    intrinsics_real, detection_real):
    """every=2 must pick raw indices 0,2,4,6 and then drop the cut ones —
    not re-index the survivors (which would shift which frames get baked
    whenever a cut is edited)."""
    import bake_corpus as bc
    trip = _make_triplet(tmp_path / "data")
    frames_index: list[dict] = []
    K, D = intrinsics_real["fisheye"]["K"], intrinsics_real["fisheye"]["D"]
    bc._bake_chunk_into(trip, TS, tmp_path / "b2" / "x", 2, K, D, detection_real,
                        frames_index, {}, {}, {}, {}, {})
    assert sorted(f["ts"] for f in frames_index) == ["12:00:00.0", "12:00:06.0"]


# ---------------------------------------------------------------------------
# build_features + audit planner
# ---------------------------------------------------------------------------

def test_build_features_drops_cut_frames_keeps_raw_index(tmp_path, curation_file,
                                                         intrinsics_real,
                                                         detection_real):
    from scripts.fusion_model.build_features import build_for_triplet
    trip = _make_triplet(tmp_path / "data")
    _, frames_df, meta = build_for_triplet(trip, intrinsics_real, detection_real,
                                           use_imu=False)
    assert meta["n_frames"] == 4
    assert sorted(frames_df["frame_index"].tolist()) == [0, 1, 5, 6]
    _, raw_df, _ = build_for_triplet(trip, intrinsics_real, detection_real,
                                     use_imu=False, respect_curation=False)
    assert len(raw_df) == N_FRAMES


def test_build_features_refuses_deleted_set(tmp_path, curation_file, intrinsics_real,
                                            detection_real):
    from scripts.fusion_model.build_features import build_for_triplet
    trip = _make_triplet(tmp_path / "data", ts="2099-07-01_11-05-00")
    with pytest.raises(FileNotFoundError, match="deleted"):
        build_for_triplet(trip, intrinsics_real, detection_real, use_imu=False)
    _, df, _ = build_for_triplet(trip, intrinsics_real, detection_real,
                                 use_imu=False, respect_curation=False)
    assert len(df) == N_FRAMES


def _sector_line(ts: str) -> str:
    return json.dumps({"timestamp": ts, "p_obstacle": [0.5, 0.5],
                       "bin_centers_deg": [-7.5, 7.5]})


def test_audit_frame_selector_honours_curation(tmp_path, curation_file):
    from scripts.eval.audit_frame_selector import main
    sectors = tmp_path / "sectors"
    sectors.mkdir()
    (sectors / f"{SCENE}__{TS}.jsonl").write_text(
        "\n".join(_sector_line(f"12:00:{i:02d}.0") for i in range(N_FRAMES)) + "\n")
    (sectors / f"{SCENE}__2099-07-01_11-05-00.jsonl").write_text(
        "\n".join(_sector_line(f"11:05:{i:02d}.0") for i in range(N_FRAMES)) + "\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    out = tmp_path / "plan.csv"
    assert main(["--sectors", str(sectors), "--bundle", str(bundle),
                 "--labels", "--n", "50", "--out", str(out)]) == 0
    rows = out.read_text().splitlines()[1:]
    clips = {r.split(",")[1] for r in rows}
    tss = {r.split(",")[2] for r in rows}
    assert clips == {f"{SCENE}__{TS}"}
    assert not tss & {"12:00:02.0", "12:00:03.0", "12:00:04.0", "12:00:07.0"}
    out2 = tmp_path / "plan_all.csv"
    assert main(["--sectors", str(sectors), "--bundle", str(bundle),
                 "--labels", "--n", "50", "--out", str(out2),
                 "--ignore-curation"]) == 0
    clips2 = {r.split(",")[1] for r in out2.read_text().splitlines()[1:]}
    assert f"{SCENE}__2099-07-01_11-05-00" in clips2
