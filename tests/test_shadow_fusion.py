"""Shadow fusion service (Phase 4): replay bench, tail sources, scorer
degradation, seg-worker adapter, CLI, laptop-safety.

No SSD data: triplets are fabricated in tmp_path (conftest patterns).
Timing is PRINTED by the bench, never asserted (CI machines vary)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.data_collection.shadow_fusion import (
    DEFAULT_SEG_MAX_AGE_S,
    ReplaySources,
    SegWorkerProvider,
    ShadowService,
    TailSources,
    _CsvTail,
    main,
)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run_replay(tmp_path, intrinsics, detection, make_triplet, ts,
                scorer_path=None, seg_provider=None, n=5):
    trip = make_triplet(tmp_path, ts, n=n)
    src = ReplaySources(str(trip.fisheye.parent / trip.timestamp), detection)
    out = tmp_path / f"shadow_{ts}.jsonl"
    svc = ShadowService(src, out, intrinsics=intrinsics, detection=detection,
                        scorer_path=scorer_path, seg_provider=seg_provider)
    stats = svc.run()
    return out, stats


# ---------------------------------------------------------------------------
# Replay bench path
# ---------------------------------------------------------------------------

def test_replay_emits_one_record_per_frame(tmp_path, intrinsics_real,
                                           detection_real,
                                           make_fusion_triplet):
    out, stats = _run_replay(tmp_path, intrinsics_real, detection_real,
                             make_fusion_triplet, "2099-07-01_12-00-00", n=5)
    recs = _read_records(out)
    assert len(recs) == stats.n_records == 5

    f = detection_real["fusion"]
    n_bins = len(np.arange(f["bin_min_deg"], f["bin_max_deg"],
                           f["bin_step_deg"]))
    for rec in recs:
        assert len(rec["bin_centers_deg"]) == n_bins
        assert len(rec["scores"]) == n_bins
        assert len(rec["min_range_m"]) == n_bins
        assert len(rec["sensor_hits"]) == n_bins
        # additive health block: fixed keys, never renamed
        sh = rec["shadow"]
        assert set(sh) >= {"tick_ms", "seg_age_s", "seg_fresh", "scorer_ok"}
        assert isinstance(sh["tick_ms"], float) and sh["tick_ms"] >= 0.0
        assert sh["scorer_ok"] is False          # no scorer configured
        assert "p_obstacle" not in rec
    # replay timeline discipline: timestamps are the triplet's own
    # (mmwave RoundedTime), never the wall clock
    assert recs[0]["timestamp"] == "12:00:00.0"
    assert recs[1]["timestamp"] == "12:00:01.0"
    # the synthetic radar point (0.1, 1.5) must register as a radar hit
    assert any(h[2] for rec in recs for h in rec["sensor_hits"])
    assert len(stats.tick_ms) == 5


def test_replay_pipeline_is_stateful_single_instance(tmp_path,
                                                     intrinsics_real,
                                                     detection_real,
                                                     make_fusion_triplet):
    """One pipeline for the service lifetime: the thermal running average
    must evolve across frames (not reset per tick)."""
    trip = make_fusion_triplet(tmp_path, "2099-07-01_13-00-00", n=4)
    src = ReplaySources(str(trip.fisheye.parent / trip.timestamp),
                        detection_real)
    svc = ShadowService(src, tmp_path / "o.jsonl", intrinsics=intrinsics_real,
                        detection=detection_real)
    avg0 = svc.pipeline.thermal_average
    svc.run()
    assert svc.pipeline.thermal_average != avg0


# ---------------------------------------------------------------------------
# Scorer: live artefact path + degradation
# ---------------------------------------------------------------------------

def _toy_v1b_bundle(tmp_path):
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    import pandas as pd

    from scripts.fusion_model.models import (
        GBT_COLUMNS,
        GBTScorerBundle,
        IsotonicCalibrator,
        fit_gbt,
    )

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, len(GBT_COLUMNS))),
                      columns=GBT_COLUMNS)
    y = (df[GBT_COLUMNS[0]].to_numpy() > 0).astype(float)
    b = GBTScorerBundle(clfs=[fit_gbt(df, y, seed=0)],
                        train_info={"tag": "shadow-test"})
    b.calibrator = IsotonicCalibrator.fit(b.predict_proba_df(df), y)
    path = tmp_path / "scorer_v1b.joblib"
    b.save(path)
    return path


def test_replay_with_v1b_scorer_emits_p_obstacle(tmp_path, intrinsics_real,
                                                 detection_real,
                                                 make_fusion_triplet):
    scorer = _toy_v1b_bundle(tmp_path)
    out, stats = _run_replay(tmp_path, intrinsics_real, detection_real,
                             make_fusion_triplet, "2099-07-01_12-30-00",
                             scorer_path=scorer, n=3)
    recs = _read_records(out)
    assert stats.scorer_ok_frames == len(recs) == 3
    for rec in recs:
        assert rec["shadow"]["scorer_ok"] is True
        assert len(rec["p_obstacle"]) == len(rec["bin_centers_deg"])
        assert all(0.0 <= p <= 1.0 for p in rec["p_obstacle"])
        assert len(rec["threat"]) == len(rec["bin_centers_deg"])


def test_missing_scorer_degrades_to_legacy(tmp_path, intrinsics_real,
                                           detection_real,
                                           make_fusion_triplet, capsys):
    out, stats = _run_replay(tmp_path, intrinsics_real, detection_real,
                             make_fusion_triplet, "2099-07-01_14-00-00",
                             scorer_path=tmp_path / "nope.joblib", n=3)
    recs = _read_records(out)
    assert len(recs) == 3                      # never crashes the service
    assert all(r["shadow"]["scorer_ok"] is False for r in recs)
    assert all("p_obstacle" not in r for r in recs)
    assert "WARN scorer unavailable" in capsys.readouterr().out


def test_broken_scorer_disabled_after_consecutive_failures(
        tmp_path, intrinsics_real, detection_real, make_fusion_triplet,
        capsys):
    scorer = _toy_v1b_bundle(tmp_path)
    trip = make_fusion_triplet(tmp_path, "2099-07-01_15-00-00", n=5)
    src = ReplaySources(str(trip.fisheye.parent / trip.timestamp),
                        detection_real)
    svc = ShadowService(src, tmp_path / "o.jsonl", intrinsics=intrinsics_real,
                        detection=detection_real, scorer_path=scorer)

    class _Boom:
        def predict_proba_calibrated_df(self, df):
            raise RuntimeError("boom")

    svc._sc["scorer"] = _Boom()                 # break the per-frame path
    stats = svc.run()
    assert stats.n_records == 5                 # service survives
    assert stats.scorer_ok_frames == 0
    assert svc._scorer is None                  # disabled after max failures
    assert "scorer disabled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tail mode (beside capture): growing CSVs, read-only, no camera pixels
# ---------------------------------------------------------------------------

TS = "2099-01-01_00-00-00"


def _capture_writer(sess: Path, n: int, period_s: float = 0.05):
    """Simulate the capture service appending rows (flushed per row so the
    test isn't at the mercy of io buffering; the real service's buffering is
    a documented latency, not a correctness issue)."""
    frames = sess / f"frames_{TS}.csv"
    mm = sess / f"mmwave_{TS}.csv"
    with open(frames, "w") as ffh, open(mm, "w") as mfh:
        ffh.write("frame_index,Date,Time,ExposureTime,AnalogueGain,"
                  "DigitalGain,Lux\n")
        mfh.write("Date,Time,X,Y,Z,V,SNR,NOISE\n")
        ffh.flush()
        mfh.flush()
        for i in range(n):
            t = f"00:00:{i:02d}.0"
            # radar group first (2 points), then the frame row for the tick
            mfh.write(f"2099-01-01,{t},0.1,1.5,0.0,0.2,14.0,8.0\n")
            mfh.write(f"2099-01-01,{t},0.5,2.0,0.0,0.1,12.0,8.0\n")
            mfh.flush()
            ffh.write(f"{i},2099-01-01,{t},333333,8.0,1.0,4.0\n")
            ffh.flush()
            time.sleep(period_s)


def test_tail_sources_beside_capture(tmp_path, intrinsics_real,
                                     detection_real):
    sess = tmp_path / TS
    sess.mkdir()
    writer = threading.Thread(target=_capture_writer, args=(sess, 8),
                              daemon=True)
    writer.start()
    src = TailSources(sess, idle_timeout_s=3.0)
    out = sess / "shadow_test.jsonl"
    svc = ShadowService(src, out, intrinsics=intrinsics_real,
                        detection=detection_real)
    stats = svc.run(max_frames=5)
    writer.join(timeout=5)
    recs = _read_records(out)
    assert len(recs) == 5
    for rec in recs:
        # no camera pixels beside capture: honest absence, never fabricated
        hits = np.asarray(rec["sensor_hits"])
        assert not hits[:, 0].any() and not hits[:, 1].any()
        assert rec["shadow"]["source_latency_s"] >= 0.0
    # radar evidence flows once >= 2 timestamp groups exist (the newest
    # group may still be mid-write, so the tick uses the previous one)
    assert any(np.asarray(r["sensor_hits"])[:, 2].any() for r in recs[1:])
    # tail timeline discipline: capture's own stamps, not the wall clock
    assert recs[0]["timestamp"].startswith("00:00:0")
    # read-only contract: shadow created ONLY its own output in the session
    created = {p.name for p in sess.iterdir()}
    assert created == {f"frames_{TS}.csv", f"mmwave_{TS}.csv",
                       "shadow_test.jsonl"}


def test_tail_sources_idle_timeout_stops(tmp_path, intrinsics_real,
                                         detection_real):
    sess = tmp_path / TS
    sess.mkdir()
    _capture_writer(sess, 2, period_s=0.0)      # pre-written, then silence
    src = TailSources(sess, idle_timeout_s=0.5)
    svc = ShadowService(src, sess / "s.jsonl", intrinsics=intrinsics_real,
                        detection=detection_real)
    stats = svc.run()                            # must return, not hang
    assert stats.n_records == 2


def test_csv_tail_partial_lines(tmp_path):
    p = tmp_path / "grow.csv"
    p.write_text("A,B\n1,2\n3,")
    tail = _CsvTail(p)
    assert tail.read_new_rows() == [["1", "2"]]   # torn line withheld
    with open(p, "a") as fh:
        fh.write("4\n5,6\n")
    assert tail.read_new_rows() == [["3", "4"], ["5", "6"]]


# ---------------------------------------------------------------------------
# Seg worker adapter (fake worker: no onnxruntime in tests)
# ---------------------------------------------------------------------------

class _FakeSegResult:
    def __init__(self, mask, age_s):
        self.mask = mask
        self._age = age_s

    def age_s(self, now=None):
        return self._age


class _FakeSegWorker:
    def __init__(self, mask, age_s=0.5):
        self.result = _FakeSegResult(mask, age_s)
        self.submits = []

    def latest(self, max_age_s=None):
        return self.result

    def submit(self, frame, ts=None):
        self.submits.append(frame)

    def stop(self):
        pass


def test_seg_provider_fresh_vs_stale():
    mask = np.ones((120, 160), dtype=np.uint8)
    fresh = SegWorkerProvider(_FakeSegWorker(mask, age_s=0.5),
                              max_age_s=DEFAULT_SEG_MAX_AGE_S)
    assert fresh.get(None) is mask
    assert fresh.last_age_s == 0.5
    stale = SegWorkerProvider(_FakeSegWorker(mask, age_s=5.0),
                              max_age_s=DEFAULT_SEG_MAX_AGE_S)
    assert stale.get(None) is None               # degrade to classical path
    assert stale.last_age_s == 5.0


def test_replay_with_fake_seg_worker(tmp_path, intrinsics_real,
                                     detection_real, make_fusion_triplet):
    mask = np.ones((120, 160), dtype=np.uint8)   # all-water (SEG classes)
    worker = _FakeSegWorker(mask, age_s=0.3)
    provider = SegWorkerProvider(worker)
    out, stats = _run_replay(tmp_path, intrinsics_real, detection_real,
                             make_fusion_triplet, "2099-07-01_16-00-00",
                             seg_provider=provider, n=3)
    recs = _read_records(out)
    assert stats.seg_fresh_frames == 3
    for rec in recs:
        assert rec["shadow"]["seg_fresh"] is True
        assert rec["shadow"]["seg_age_s"] == 0.3
    # every frame's undistorted fisheye was offered to the async worker
    assert len(worker.submits) == 3
    assert worker.submits[0].shape == (120, 160, 3)


# ---------------------------------------------------------------------------
# CLI + laptop safety
# ---------------------------------------------------------------------------

def test_cli_replay_bench(tmp_path, make_fusion_triplet, capsys):
    trip = make_fusion_triplet(tmp_path, "2099-07-02_12-00-00", n=4)
    out = tmp_path / "cli.jsonl"
    rc = main(["--replay", str(trip.fisheye.parent / trip.timestamp),
               "--out", str(out), "--max-frames", "2", "--scorer", "none"])
    assert rc == 0
    assert len(_read_records(out)) == 2
    printed = capsys.readouterr().out
    assert "tick p50" in printed and "p95" in printed   # measured, not asserted


def test_live_refuses_cleanly_without_pi_hardware():
    from scripts.data_collection import continuous_capture as cc
    if cc.Picamera2 is not None:
        pytest.skip("running on a Pi with picamera2: live mode is real here")
    with pytest.raises(SystemExit) as ei:
        main(["--live", "--scorer", "none"])
    assert "picamera2" in str(ei.value)


def test_module_imports_without_pi_or_gpu_deps():
    import scripts.data_collection.shadow_fusion as sf
    assert sf.TICK_BUDGET_MS == 333.0


def test_tail_sources_with_snapshot_carries_camera_and_latest_wins(
        tmp_path, intrinsics_real, detection_real):
    """With capture's frame snapshot the tail tick has camera pixels (joined
    by (chunk, frame_index), never by wall-clock) and processes only the
    newest sidecar row per tick when it falls behind."""
    from scripts.data_collection.frame_snapshot import SnapshotWriter
    sess = tmp_path / TS
    sess.mkdir()
    snap_dir = tmp_path / "shm"
    n_rows = 8
    fisheye = np.full((648, 864, 3), 90, dtype=np.uint8)

    def writer_with_snapshots():
        frames = sess / f"frames_{TS}.csv"
        mm = sess / f"mmwave_{TS}.csv"
        snap = SnapshotWriter(snap_dir)
        with open(frames, "w") as ffh, open(mm, "w") as mfh:
            ffh.write("frame_index,Date,Time,ExposureTime,AnalogueGain,"
                      "DigitalGain,Lux\n")
            mfh.write("Date,Time,X,Y,Z,V,SNR,NOISE\n")
            ffh.flush(); mfh.flush()
            for i in range(n_rows):
                t = f"00:00:{i:02d}.0"
                mfh.write(f"2099-01-01,{t},0.1,1.5,0.0,0.2,14.0,8.0\n")
                mfh.flush()
                snap.publish(TS, i, t, fisheye, None)
                ffh.write(f"{i},2099-01-01,{t},333333,8.0,1.0,4.0\n")
                ffh.flush()
                time.sleep(0.05)

    # write everything BEFORE the tail starts: all rows arrive in one burst,
    # so latest-wins must process only the newest one and skip the rest
    writer_with_snapshots()
    src = TailSources(sess, idle_timeout_s=1.0, snapshot_dir=snap_dir)
    out = sess / "shadow_test.jsonl"
    svc = ShadowService(src, out, intrinsics=intrinsics_real,
                        detection=detection_real)
    svc.run(max_frames=3)
    recs = _read_records(out)
    assert len(recs) == 1, "burst of rows -> one latest-wins tick"
    rec = recs[0]
    assert rec["timestamp"] == f"00:00:{n_rows - 1:02d}.0"
    assert rec["shadow"]["camera"] is True
    assert rec["shadow"]["skipped_rows"] == n_rows - 1
    assert src.camera_frames == 1 and src.camera_misses == 0
    # without a snapshot dir the same session is camera-blind and processes
    # every row (unchanged behaviour)
    src2 = TailSources(sess, idle_timeout_s=1.0)
    out2 = sess / "shadow_blind.jsonl"
    ShadowService(src2, out2, intrinsics=intrinsics_real,
                  detection=detection_real).run(max_frames=3)
    recs2 = _read_records(out2)
    assert len(recs2) == 3 and "camera" not in recs2[0]["shadow"]
    assert not np.asarray(recs2[0]["sensor_hits"])[:, 0].any()


def test_tail_snapshot_for_another_frame_is_not_used(tmp_path, intrinsics_real,
                                                     detection_real):
    from scripts.data_collection.frame_snapshot import SnapshotWriter
    sess = tmp_path / TS
    sess.mkdir()
    snap_dir = tmp_path / "shm"
    SnapshotWriter(snap_dir).publish(TS, 99, "00:00:99.0",
                                     np.zeros((648, 864, 3), np.uint8), None)
    frames = sess / f"frames_{TS}.csv"
    frames.write_text("frame_index,Date,Time,ExposureTime,AnalogueGain,"
                      "DigitalGain,Lux\n0,2099-01-01,00:00:00.0,1,1,1,1\n")
    src = TailSources(sess, idle_timeout_s=0.5, snapshot_dir=snap_dir)
    out = sess / "shadow_test.jsonl"
    ShadowService(src, out, intrinsics=intrinsics_real,
                  detection=detection_real).run(max_frames=1)
    rec = _read_records(out)[0]
    assert rec["shadow"]["camera"] is False and src.camera_misses == 1
