import pytest

import scripts.utils.datasets as ds
from scripts.utils.datasets import (
    Triplet,
    list_capture_missions,
    list_triplets,
    load_mmwave_csv,
    resolve_triplet,
)


def test_resolve_triplet_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_triplet(tmp_path / "Scene" / "1999-01-01_00-00-00")


def test_resolve_triplet_real(boats_triplet):
    assert boats_triplet.scene == "Boats"
    assert boats_triplet.fisheye.exists()
    assert boats_triplet.thermal.exists()
    assert boats_triplet.mmwave.exists()
    assert boats_triplet.clip_id == f"Boats/{boats_triplet.timestamp}"


@pytest.mark.needs_data
def test_list_triplets_finds_at_least_one_per_scene():
    triplets = list_triplets()
    assert len(triplets) >= 4
    scenes = {t.scene for t in triplets}
    assert scenes >= {"Boats", "Ducks", "OpenWater", "Rain"}


@pytest.mark.needs_data
def test_list_triplets_scene_filter():
    boats = list_triplets("Boats")
    assert all(t.scene == "Boats" for t in boats)
    assert len(boats) >= 1


def test_list_triplets_nonexistent_scene_returns_empty():
    assert list_triplets("Nonexistent") == []


def test_load_mmwave_csv_adds_rounded_time(boats_triplet):
    df = load_mmwave_csv(boats_triplet.mmwave)
    assert "RoundedTime" in df.columns
    assert "Timestamp" in df.columns
    assert df["RoundedTime"].str.len().eq(10).all()  # HH:MM:SS.f
    for col in ("X", "Y", "Z"):
        assert col in df.columns


def test_triplet_dataclass_is_frozen():
    t = Triplet(scene="X", timestamp="ts",
                fisheye=pytest.importorskip("pathlib").Path("/x"),
                thermal=pytest.importorskip("pathlib").Path("/x"),
                mmwave=pytest.importorskip("pathlib").Path("/x"))
    with pytest.raises(Exception):
        t.scene = "Y"  # frozen


def test_load_mmwave_csv_drops_garbage_rows(tmp_path):
    """Rows with absurd X/Y/Z values (from misaligned float32 reads) are dropped."""
    p = tmp_path / "mmwave_x.csv"
    p.write_text(
        "Date,Time,X,Y,Z\n"
        "2025-01-01,00:00:00.0,0.1,1.5,0.0\n"
        "2025-01-01,00:00:00.0,1.0e35,2.0,0.0\n"
        "2025-01-01,00:00:00.0,0.3,-5.0e36,0.1\n"
        "2025-01-01,00:00:00.0,2.0,3.0,0.5\n"
    )
    df = load_mmwave_csv(p)
    assert len(df) == 2  # only the two physically plausible rows


def test_load_mmwave_csv_custom_bound(tmp_path):
    p = tmp_path / "mmwave_x.csv"
    p.write_text(
        "Date,Time,X,Y,Z\n"
        "2025-01-01,00:00:00.0,0.1,1.5,0.0\n"
        "2025-01-01,00:00:00.0,30.0,2.0,0.0\n"
    )
    df = load_mmwave_csv(p, sanity_bound_m=10.0)
    assert len(df) == 1


def test_load_mmwave_csv_no_sanity_keeps_garbage(tmp_path):
    p = tmp_path / "mmwave_x.csv"
    p.write_text(
        "Date,Time,X,Y,Z\n"
        "2025-01-01,00:00:00.0,1.0e35,2.0,0.0\n"
    )
    df = load_mmwave_csv(p, sanity_bound_m=None)
    assert len(df) == 1


def test_list_capture_missions_absent_dir_returns_empty(tmp_path, monkeypatch):
    """When data/captures/ does not exist, return [] (line 31)."""
    monkeypatch.setattr(ds, "CAPTURES_DIR", tmp_path / "no_such_captures")
    assert list_capture_missions() == []


def test_list_capture_missions_lists_dirs(tmp_path, monkeypatch):
    cap = tmp_path / "captures"
    (cap / "missionB").mkdir(parents=True)
    (cap / "missionA").mkdir(parents=True)
    (cap / "afile.txt").write_text("x")  # files ignored
    monkeypatch.setattr(ds, "CAPTURES_DIR", cap)
    assert list_capture_missions() == ["missionA", "missionB"]


def test_list_triplets_skips_incomplete(tmp_path, monkeypatch):
    """A scene with a lone fisheye mp4 (no thermal/mmwave) is skipped via the
    FileNotFoundError continue branch (lines 90-91)."""
    scene_dir = tmp_path / "Boats"
    scene_dir.mkdir(parents=True)
    # Only the fisheye file exists -> resolve_triplet raises -> skipped.
    (scene_dir / "fisheye_2030-01-01_00-00-00.mp4").write_bytes(b"x")
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ds, "CAPTURES_DIR", tmp_path / "captures")
    assert list_triplets("Boats") == []


def test_resolve_triplet_nested_mission_scene_flattens(tmp_path, monkeypatch):
    """data/captures/<mission>/<sub>/<ts> resolves with scene
    `<mission>_<sub>` so frame_ids match the flattened artefact naming
    data/seg + data/det_seg use (e.g. 2026-08-19_afloat_session__<ts>)."""
    cap = tmp_path / "captures"
    clip = cap / "2026-08-19_afloat" / "session"
    clip.mkdir(parents=True)
    ts = "2026-08-19_16-21-09"
    for name in (f"fisheye_{ts}.mp4", f"thermal_{ts}.mp4", f"mmwave_{ts}.csv"):
        (clip / name).write_bytes(b"x")
    monkeypatch.setattr(ds, "CAPTURES_DIR", cap)
    t = resolve_triplet(clip / ts)
    assert t.scene == "2026-08-19_afloat_session"
    assert t.clip_id == f"2026-08-19_afloat_session/{ts}"
    # Single-level missions keep the mission name verbatim.
    flat = cap / "2026-07-08"
    flat.mkdir()
    ts2 = "2026-07-08_16-37-01"
    for name in (f"fisheye_{ts2}.mp4", f"thermal_{ts2}.mp4",
                 f"mmwave_{ts2}.csv"):
        (flat / name).write_bytes(b"x")
    assert resolve_triplet(flat / ts2).scene == "2026-07-08"


def test_load_mmwave_csv_empty_file(tmp_path):
    """A header-less empty CSV triggers EmptyDataError -> empty frame
    fallback (lines 122-123)."""
    p = tmp_path / "mmwave_empty.csv"
    p.write_text("")
    df = load_mmwave_csv(p)
    assert df.empty
    assert "RoundedTime" in df.columns
    assert "Timestamp" in df.columns


# ---------------------------------------------------------------------------
# Optional V (Doppler) column: captured since 2026-07-06
# ---------------------------------------------------------------------------

def test_load_mmwave_csv_without_v_column_unchanged(tmp_path):
    """Pre-Doppler CSVs must load exactly as before (no V column appears)."""
    p = tmp_path / "mmwave_old.csv"
    p.write_text(
        "Date,Time,X,Y,Z\n"
        "2026-07-07,10:00:00.0,0.1,1.5,0.0\n"
    )
    df = load_mmwave_csv(p)
    assert "V" not in df.columns
    assert len(df) == 1


def test_load_mmwave_csv_with_v_column(tmp_path):
    """Point rows keep their Doppler; sentinel rows carry NaN V."""
    import numpy as np
    p = tmp_path / "mmwave_v.csv"
    p.write_text(
        "Date,Time,X,Y,Z,V\n"
        "2026-07-07,10:00:00.0,0.1,1.5,0.0,-0.8\n"
        "2026-07-07,10:00:00.1,,,,\n"           # sentinel heartbeat
        "2026-07-07,10:00:00.2,0.2,2.0,0.0,0.4\n"
    )
    df = load_mmwave_csv(p)
    assert "V" in df.columns
    assert len(df) == 3
    assert df["V"].iloc[0] == pytest.approx(-0.8)
    assert np.isnan(df["V"].iloc[1])
    assert df["V"].iloc[2] == pytest.approx(0.4)


def test_load_mmwave_csv_implausible_doppler_nans_not_dropped(tmp_path):
    """|V| beyond the Doppler sanity bound is masked to NaN, but the row
    (whose position is plausible) survives."""
    import numpy as np
    p = tmp_path / "mmwave_v_garbage.csv"
    p.write_text(
        "Date,Time,X,Y,Z,V\n"
        "2026-07-07,10:00:00.0,0.1,1.5,0.0,1.0e30\n"
        "2026-07-07,10:00:00.0,0.2,2.0,0.0,-0.5\n"
    )
    df = load_mmwave_csv(p)
    assert len(df) == 2
    assert np.isnan(df["V"].iloc[0])
    assert df["V"].iloc[1] == pytest.approx(-0.5)


# --- fisheye-only captures --------------------------------------------------

def _write_fisheye_clip(scene_dir, ts, n=4):
    """The only files a ASVPROJECT_FISHEYE_ONLY capture leaves on disk."""
    import cv2
    import numpy as np

    scene_dir.mkdir(parents=True, exist_ok=True)
    fish = scene_dir / f"fisheye_{ts}.mp4"
    fw = cv2.VideoWriter(str(fish), cv2.VideoWriter_fourcc(*"mp4v"), 3.0,
                         (160, 120))
    for _ in range(n):
        fw.write(np.full((120, 160, 3), 70, dtype=np.uint8))
    fw.release()
    (scene_dir / f"frames_{ts}.csv").write_text(
        "frame_index,Date,Time\n0,2099-01-01,00:00:00.0\n")
    return fish


def test_resolve_triplet_strict_by_default_rejects_fisheye_only(tmp_path):
    """The default must stay strict: the fusion pipeline has to keep failing
    loudly on an incomplete clip rather than silently fusing one sensor."""
    _write_fisheye_clip(tmp_path / "Synth", "2099-01-01_00-00-00")
    with pytest.raises(FileNotFoundError):
        resolve_triplet(tmp_path / "Synth" / "2099-01-01_00-00-00")


def test_resolve_triplet_fisheye_only_when_relaxed(tmp_path):
    _write_fisheye_clip(tmp_path / "Synth", "2099-01-01_00-00-00")
    t = resolve_triplet(tmp_path / "Synth" / "2099-01-01_00-00-00",
                        require=("fisheye",))
    assert t.fisheye.exists()
    assert t.thermal is None
    assert t.mmwave is None
    assert t.frames is not None      # sidecar still picked up


def test_resolve_triplet_relaxed_still_requires_fisheye(tmp_path):
    (tmp_path / "Synth").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_triplet(tmp_path / "Synth" / "2099-01-01_00-00-00",
                        require=("fisheye",))


# --- gps_<ts>.csv position sidecar (since 2026-08-24) ------------------------

GPS_HEADER = "Date,Time,Lat,Lon,Fix,Source,FixTime\n"


def test_load_gps_csv_reads_position_and_provenance(tmp_path):
    path = tmp_path / "gps_2026-08-24_12-00-00.csv"
    path.write_text(GPS_HEADER
                    + "2026-08-24,12:00:00.1,46.0000000,9.0000000,4,boat_log,"
                      "2026-08-24T09:59:58+00:00\n"
                    + "2026-08-24,12:30:00.4,46.0010000,9.0010000,4,boat_log,"
                      "2026-08-24T10:29:58+00:00\n")
    df = ds.load_gps_csv(path)
    assert len(df) == 2
    assert df["Lat"].iloc[0] == pytest.approx(46.000000)
    assert list(df["Source"]) == ["boat_log", "boat_log"]
    # RoundedTime is the 100 ms convention every other loader uses, so a
    # position can be matched to the radar-keyed frame timestamps.
    assert df["RoundedTime"].iloc[0] == "12:00:00.1"
    assert str(df["Timestamp"].iloc[1]).startswith("2026-08-24 12:30:00")


def test_load_gps_csv_keeps_the_fallback_marker(tmp_path):
    """`fallback` is the fixed position from configs/gps.yaml, not a
    measurement, and nothing downstream can tell from the coordinates alone."""
    path = tmp_path / "gps_x.csv"
    path.write_text(GPS_HEADER
                    + "2026-08-24,12:00:00.1,46.0000000,9.0000000,,fallback,\n")
    df = ds.load_gps_csv(path)
    assert df["Source"].iloc[0] == "fallback"


def test_load_gps_csv_empty_and_missing_files(tmp_path):
    header_only = tmp_path / "gps_empty.csv"
    header_only.write_text(GPS_HEADER)
    for path in (header_only, tmp_path / "absent.csv"):
        df = ds.load_gps_csv(path)
        assert df.empty
        assert {"Lat", "Lon", "Source", "Timestamp", "RoundedTime"} <= set(df.columns)


def test_resolve_triplet_attaches_gps_sidecar_when_present(tmp_path):
    scene = tmp_path / "Scene"
    scene.mkdir()
    ts = "2026-08-24_12-00-00"
    for name in (f"fisheye_{ts}.mp4", f"thermal_{ts}.mp4", f"mmwave_{ts}.csv"):
        (scene / name).write_text("x")
    assert resolve_triplet(scene / ts).gps is None      # absent -> None
    (scene / f"gps_{ts}.csv").write_text(GPS_HEADER)
    assert resolve_triplet(scene / ts).gps.name == f"gps_{ts}.csv"
