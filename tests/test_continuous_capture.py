"""Unit tests for the pure radar-parsing functions in
scripts/data_collection/continuous_capture.py.

The module guards its Pi-only picamera2 import, so it is importable on
the laptop; only the TLV parsing layer is exercised here (the capture
loop needs real sensors and is smoke-tested on the Pi instead:
docs/radar_diagnosis.tex verification log).

Wire format under test (TI mmWave SDK 3.x OOB demo):
  TLV 1  detected points   16 bytes/object: float32 x, y, z, v
  TLV 7  point side info    4 bytes/object: int16 snr, int16 noise
                              (both in 0.1 dB steps)
"""

import struct

import numpy as np
import pandas as pd
import pytest

from scripts.data_collection.continuous_capture import IMUReaderRVC, parse_tlvs
from scripts.utils.datasets import load_mmwave_csv


def _tlv(tlv_type: int, payload: bytes) -> bytes:
    return struct.pack("<II", tlv_type, len(payload)) + payload


def _rvc_frame(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, ax=0, ay=0, az=0, index=0):
    """Build a valid 19-byte UART-RVC frame (checksum computed)."""
    body = struct.pack(
        "<BhhhhhhBBB", index & 0xFF,
        round(yaw_deg * 100), round(pitch_deg * 100), round(roll_deg * 100),
        ax, ay, az, 0, 0, 0)
    return b"\xaa\xaa" + body + bytes([sum(body) & 0xFF])


def _points_payload(points):
    """points = [(x, y, z, v), ...] → TLV-1 payload bytes."""
    return b"".join(struct.pack("<4f", *p) for p in points)


def _side_payload(side):
    """side = [(snr_raw, noise_raw), ...] in 0.1 dB units → TLV-7 payload."""
    return b"".join(struct.pack("<hh", *s) for s in side)


POINTS = [(0.5, 2.0, 0.1, 0.0), (-1.0, 4.0, -0.2, 0.3)]
SIDE_RAW = [(123, 240), (87, 251)]        # 12.3/24.0 dB, 8.7/25.1 dB


def test_points_only_returns_none_side_info():
    data = _tlv(1, _points_payload(POINTS))
    xs, ys, zs, vs, snrs, noises = parse_tlvs(data, 1, len(POINTS))
    assert xs == [p[0] for p in POINTS]
    assert vs == pytest.approx([p[3] for p in POINTS])
    assert snrs is None and noises is None


def test_points_plus_side_info_decoded_in_db():
    data = (_tlv(1, _points_payload(POINTS))
            + _tlv(7, _side_payload(SIDE_RAW)))
    xs, ys, zs, vs, snrs, noises = parse_tlvs(data, 2, len(POINTS))
    assert len(xs) == len(snrs) == len(noises) == len(POINTS)
    assert snrs == pytest.approx([12.3, 8.7])
    assert noises == pytest.approx([24.0, 25.1])


def test_side_info_before_points_is_order_independent():
    data = (_tlv(7, _side_payload(SIDE_RAW))
            + _tlv(1, _points_payload(POINTS)))
    xs, ys, zs, vs, snrs, noises = parse_tlvs(data, 2, len(POINTS))
    assert xs == [p[0] for p in POINTS]
    assert snrs == pytest.approx([12.3, 8.7])


def test_unknown_tlvs_are_skipped():
    # TLV 2 (range profile) and TLV 6 (stats) surround the point TLVs,
    # as in the real stream (guiMonitor -1 1 1 0 0 0 1).
    data = (_tlv(2, b"\x00" * 512)
            + _tlv(1, _points_payload(POINTS))
            + _tlv(6, b"\x00" * 24)
            + _tlv(7, _side_payload(SIDE_RAW)))
    xs, ys, zs, vs, snrs, noises = parse_tlvs(data, 4, len(POINTS))
    assert ys == [p[1] for p in POINTS]
    assert snrs == pytest.approx([12.3, 8.7])


def test_spliced_point_tlv_drops_whole_frame():
    # TLV-1 length disagrees with the header's num_det_obj → spliced
    # frame → the parser must refuse to fabricate points.
    data = _tlv(1, _points_payload(POINTS))
    assert parse_tlvs(data, 1, len(POINTS) + 1) is None


def test_malformed_side_info_keeps_points():
    # A bad TLV-7 (wrong length) only discards the side info; TLV-1 is
    # the splice detector and the points remain trustworthy.
    data = (_tlv(1, _points_payload(POINTS))
            + _tlv(7, _side_payload(SIDE_RAW)[:-2]))
    xs, ys, zs, vs, snrs, noises = parse_tlvs(data, 2, len(POINTS))
    assert xs == [p[0] for p in POINTS]
    assert snrs is None and noises is None


def test_truncated_tlv_walk_returns_none():
    data = _tlv(1, _points_payload(POINTS))[:-4]
    assert parse_tlvs(data, 1, len(POINTS)) is None


def test_negative_snr_raw_decodes_signed():
    # int16, not uint16: the noise field in particular is documented as
    # signed in the SDK struct (DPIF_PointCloudSideInfo).
    data = (_tlv(1, _points_payload(POINTS[:1]))
            + _tlv(7, _side_payload([(-5, -10)])))
    *_, snrs, noises = parse_tlvs(data, 2, 1)
    assert snrs == pytest.approx([-0.5])
    assert noises == pytest.approx([-1.0])


# --- loader side (scripts.utils.datasets.load_mmwave_csv) -----------------

def test_loader_reads_snr_noise_columns(tmp_path):
    p = tmp_path / "mmwave_2026-07-09_12-00-00.csv"
    p.write_text(
        "Date,Time,X,Y,Z,V,SNR,NOISE\n"
        "2026-07-09,12:00:00.1,0.5,2.0,0.1,0.0,12.3,24.0\n"
        "2026-07-09,12:00:00.2,,,,,,\n"          # sentinel row
        "2026-07-09,12:00:00.3,1.0,3.0,0.0,0.1,8.7,25.1\n")
    df = load_mmwave_csv(p)
    assert list(df["SNR"].dropna()) == pytest.approx([12.3, 8.7])
    assert list(df["NOISE"].dropna()) == pytest.approx([24.0, 25.1])
    assert df["SNR"].isna().sum() == 1               # sentinel → NaN
    assert pd.api.types.is_numeric_dtype(df["SNR"])


def test_loader_without_snr_columns_unchanged(tmp_path):
    p = tmp_path / "mmwave_2026-07-06_12-00-00.csv"
    p.write_text(
        "Date,Time,X,Y,Z,V\n"
        "2026-07-06,12:00:00.1,0.5,2.0,0.1,0.0\n")
    df = load_mmwave_csv(p)
    assert "SNR" not in df.columns
    assert len(df) == 1


# --- IMU UART-RVC decode (the box IMU, default since 2026-07-14) ------------

def test_rvc_decode_roundtrip():
    yaw, pitch, roll, ax, ay, az = IMUReaderRVC._decode(
        _rvc_frame(yaw_deg=12.34, pitch_deg=-5.6, roll_deg=7.8,
                   ax=100, ay=-50, az=1000))
    assert yaw == pytest.approx(12.34, abs=0.01)
    assert pitch == pytest.approx(-5.6, abs=0.01)
    assert roll == pytest.approx(7.8, abs=0.01)
    assert az == pytest.approx(1000 * 9.80665 / 1000, abs=1e-4)


def test_rvc_decode_bad_checksum():
    f = bytearray(_rvc_frame(yaw_deg=1.0))
    f[-1] ^= 0xFF
    assert IMUReaderRVC._decode(bytes(f)) is None


def test_rvc_decode_bad_header_and_short():
    f = bytearray(_rvc_frame())
    f[1] = 0x00
    assert IMUReaderRVC._decode(bytes(f)) is None
    assert IMUReaderRVC._decode(b"\xaa\xaa\x00") is None


def test_rvc_reader_csv_interface():
    # write_one_chunk relies on these being consistent (Date,Time + columns)
    assert IMUReaderRVC.kind == "rvc"
    assert IMUReaderRVC.csv_columns == ("Yaw", "Pitch", "Roll", "Ax", "Ay", "Az")


# --- Sensor-enable flags (fisheye-only capture) ----------------------------
#
# The box is also run with the RGB fisheye as its only vision sensor (a
# downstream project, and the reduced-power mode for a supply that trips its
# over-current protection at the full-stack camera inrush). These pin the flag
# resolution and the "disabled sensor writes no file" contract, because getting
# either wrong silently produces triplets with dead legs.

_FLAG_ENV = ("ASVPROJECT_FISHEYE_ONLY", "ASVPROJECT_RADAR_ENABLE",
             "ASVPROJECT_THERMAL_ENABLE", "ASVPROJECT_IMU_ENABLE",
             "ASVPROJECT_GPS_ENABLE", "ASVPROJECT_CHUNK_SECONDS",
             "ASVPROJECT_FPS", "ASVPROJECT_THERMAL_ROTATE_DEG",
             "ASVPROJECT_BOX_CONFIG", "ASVPROJECT_SESSION_SUBDIR")


@pytest.fixture
def reload_cc(monkeypatch):
    """Re-import continuous_capture with a given environment (its flags are
    module-level, resolved at import time), then restore the pristine module so
    the reload can't leak a sensor set into any other test."""
    import importlib

    from scripts.data_collection import continuous_capture as cc

    def _reload(**env):
        for key in _FLAG_ENV:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(cc)

    yield _reload
    for key in _FLAG_ENV:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(cc)


def test_sensor_flags_default_to_the_full_stack(reload_cc):
    cc = reload_cc()
    assert (cc.FISHEYE_ONLY, cc.RADAR_ENABLE, cc.THERMAL_ENABLE,
            cc.IMU_ENABLE) == (False, True, True, True)


def test_fisheye_only_disables_radar_and_thermal_but_not_imu(reload_cc):
    cc = reload_cc(ASVPROJECT_FISHEYE_ONLY="1")
    assert cc.FISHEYE_ONLY is True
    assert cc.RADAR_ENABLE is False
    assert cc.THERMAL_ENABLE is False
    assert cc.IMU_ENABLE is True      # independent, not a vision sensor


def test_individual_sensor_flags_are_independent(reload_cc):
    cc = reload_cc(ASVPROJECT_THERMAL_ENABLE="0")
    assert cc.THERMAL_ENABLE is False
    assert cc.RADAR_ENABLE is True    # radar untouched
    assert cc.FISHEYE_ONLY is False


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", ""])
def test_env_flag_falsy_spellings(reload_cc, value):
    cc = reload_cc(ASVPROJECT_RADAR_ENABLE=value)
    assert cc.RADAR_ENABLE is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_flag_truthy_spellings(reload_cc, value):
    cc = reload_cc(ASVPROJECT_RADAR_ENABLE=value)
    assert cc.RADAR_ENABLE is True


def test_fisheye_only_chunk_writes_no_radar_or_thermal_files(
        reload_cc, monkeypatch, tmp_path):
    """A fisheye-only chunk must produce the fisheye mp4 + frames sidecar and
    NOTHING else, no empty mmwave CSV, no 44-byte thermal mp4 that would look
    like a broken triplet to `scripts.data.ingest`."""
    cc = reload_cc(ASVPROJECT_FISHEYE_ONLY="1", ASVPROJECT_CHUNK_SECONDS="0")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(None, None, None, None)

    assert list(tmp_path.glob("fisheye_*.mp4"))
    assert list(tmp_path.glob("frames_*.csv"))
    assert not list(tmp_path.glob("mmwave_*.csv"))
    assert not list(tmp_path.glob("thermal_*.mp4"))


def test_full_stack_chunk_still_records_dead_sensors(
        reload_cc, monkeypatch, tmp_path):
    """With the sensors ENABLED but absent, the empty artefacts are still
    written; that is the on-disk evidence of 'expected here, said nothing',
    and it must not be confused with the disabled case above."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="0")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(None, None, None, None)

    assert list(tmp_path.glob("mmwave_*.csv"))
    assert list(tmp_path.glob("thermal_*.mp4"))


# --- Thermal discovery must never block (ablation / missing-sensor runs) -----
#
# With the PureThermal unplugged, the discovery walk used to probe every
# /dev/video* node. The SoC's own ISP and codec blocks appear there, open
# successfully, and a read() on one can block forever: /dev/video14
# (bcm2835-isp) wedged the capture service indefinitely on SensorBox
# 2026-08-06. A missing sensor has to degrade to "absent", never hang, or an
# ablation run with a sensor removed silently records nothing.

def test_non_camera_v4l2_nodes_are_skipped(monkeypatch, reload_cc):
    cc = reload_cc()
    nodes = {
        "/dev/video0": "unicam",
        "/dev/video10": "bcm2835-codec-decode",
        "/dev/video14": "bcm2835-isp",
        "/dev/video31": "bcm2835-codec-encode_image",
        "/dev/video8": "PureThermal (fw:v1.3.0)",
    }
    monkeypatch.setattr(cc.glob, "glob", lambda pat: (
        [] if "by-id" in pat else sorted(nodes)))
    monkeypatch.setattr(cc, "_v4l2_node_name", lambda dev: nodes[dev].lower())

    probed = []

    class FakeCap:
        def __init__(self, dev, *a):
            probed.append(dev)
            self.dev = dev

        def isOpened(self):
            return True

        def set(self, *a):
            pass

        def read(self):
            # Any on-SoC node reached here would have hung the real service.
            assert "video8" in self.dev, f"probed a non-camera node: {self.dev}"
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cc.cv2, "VideoCapture", FakeCap)
    handle = cc._open_thermal()
    assert handle is not None
    assert probed == ["/dev/video8"]      # nothing else was even opened


def test_thermal_absent_returns_none_without_probing_soc_nodes(
        monkeypatch, reload_cc):
    """The ablation case: no thermal fitted at all. Discovery must return None
    promptly rather than reading an ISP node."""
    cc = reload_cc()
    nodes = {"/dev/video0": "unicam", "/dev/video14": "bcm2835-isp"}
    monkeypatch.setattr(cc.glob, "glob", lambda pat: (
        [] if "by-id" in pat else sorted(nodes)))
    monkeypatch.setattr(cc, "_v4l2_node_name", lambda dev: nodes[dev].lower())

    def boom(*a, **k):
        raise AssertionError("must not open an on-SoC node")

    monkeypatch.setattr(cc.cv2, "VideoCapture", boom)
    assert cc._open_thermal() is None


def test_v4l2_node_name_missing_sysfs_is_empty(reload_cc):
    cc = reload_cc()
    assert cc._v4l2_node_name("/dev/video-does-not-exist") == ""


# --- The first read after opening the PureThermal is unreliable -------------
#
# Measured on SensorBox 2026-08-17: a single cap.read() rejected a healthy
# Lepton on roughly half of all attempts, so the capture reported "no working
# thermal camera" and the chunk recorded no thermal at all. Retrying 3x at
# 0.3 s scored 12/12, with and without picamera2 running. The retry is limited
# to the by-id node so an unresponsive scanned node still cannot stall
# discovery (see the block above).

def _thermal_frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_thermal_by_id_node_is_retried_when_the_first_read_is_empty(
        monkeypatch, reload_cc):
    cc = reload_cc()
    monkeypatch.setattr(cc.glob, "glob", lambda pat: (
        ["/dev/v4l/by-id/usb-GroupGets_PureThermal_x-video-index0"]
        if "by-id" in pat else ["/dev/video1"]))
    monkeypatch.setattr(cc.os.path, "realpath", lambda p: "/dev/video1")
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)

    reads = []

    class FlakyCap:
        def __init__(self, dev, *a):
            self.dev = dev

        def isOpened(self):
            return True

        def set(self, *a):
            pass

        def read(self):
            reads.append(self.dev)
            if len(reads) < 3:          # first two reads come back empty
                return False, None
            return True, _thermal_frame()

        def release(self):
            pass

    monkeypatch.setattr(cc.cv2, "VideoCapture", FlakyCap)
    handle = cc._open_thermal()
    assert handle is not None, "a healthy camera was rejected on its first read"
    assert len(reads) == 3


def test_thermal_scanned_node_is_probed_only_once(monkeypatch, reload_cc):
    """No by-id symlink: the fallback scan must not multiply its reads, because
    an unresponsive node can block for seconds per read."""
    cc = reload_cc()
    monkeypatch.setattr(cc.glob, "glob", lambda pat: (
        [] if "by-id" in pat else ["/dev/video9"]))
    monkeypatch.setattr(cc, "_v4l2_node_name", lambda dev: "some-usb-camera")
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)

    reads = []

    class DeadCap:
        def __init__(self, dev, *a):
            pass

        def isOpened(self):
            return True

        def set(self, *a):
            pass

        def read(self):
            reads.append(1)
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cc.cv2, "VideoCapture", DeadCap)
    assert cc._open_thermal() is None
    assert len(reads) == 1


# --- Thermal mount rotation is applied at CAPTURE time ----------------------
#
# The orientation is a property of the BOX, not the pipeline: it has varied
# across builds, so a processing-side global default would silently re-orient
# the legacy corpus. Rotating on write keeps everything on disk canonical.

def test_thermal_rotation_defaults_to_no_op(reload_cc):
    cc = reload_cc()
    assert cc.THERMAL_ROTATE_DEG == 0
    f = np.zeros((120, 160, 3), dtype=np.uint8)
    f[0, 0] = 255                                  # top-left marker
    out = cc._rotate_thermal(f)
    assert out[0, 0].tolist() == [255, 255, 255]   # untouched


def test_thermal_rotation_180_moves_the_corner(reload_cc):
    cc = reload_cc(ASVPROJECT_THERMAL_ROTATE_DEG="180")
    assert cc.THERMAL_ROTATE_DEG == 180
    f = np.zeros((120, 160, 3), dtype=np.uint8)
    f[0, 0] = 255
    out = cc._rotate_thermal(f)
    assert out.shape == f.shape                    # 180 preserves dimensions
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[-1, -1].tolist() == [255, 255, 255]  # corner is now bottom-right


def test_thermal_rotation_90_swaps_dimensions(reload_cc):
    """A 90/270 mount must carry its swapped size, or VideoWriter silently
    drops every frame and the chunk gets an empty thermal mp4."""
    cc = reload_cc(ASVPROJECT_THERMAL_ROTATE_DEG="90")
    out = cc._rotate_thermal(np.zeros((120, 160, 3), dtype=np.uint8))
    assert out.shape[:2] == (160, 120)


def test_thermal_rotation_rejects_a_nonsense_angle(monkeypatch):
    import importlib
    from scripts.data_collection import continuous_capture as cc
    monkeypatch.setenv("ASVPROJECT_THERMAL_ROTATE_DEG", "45")
    with pytest.raises(SystemExit):
        importlib.reload(cc)
    monkeypatch.delenv("ASVPROJECT_THERMAL_ROTATE_DEG")
    importlib.reload(cc)          # restore a pristine module for other tests


# --- Per-box constants must reach manual runs, not just the service ---------
#
# ASVPROJECT_THERMAL_ROTATE_DEG lived only in the systemd unit's Environment=,
# so smoke_capture and calibration walks launched by hand recorded UNROTATED
# thermal while the service recorded rotated (measured 2026-08-19). The box
# file closes that gap.

def test_box_env_falls_back_to_the_box_file(tmp_path, reload_cc):
    box = tmp_path / "box.env"
    box.write_text("# a comment\n\nASVPROJECT_THERMAL_ROTATE_DEG=180\n")
    cc = reload_cc(ASVPROJECT_BOX_CONFIG=str(box))
    assert cc.THERMAL_ROTATE_DEG == 180


def test_box_env_environment_wins_over_the_box_file(tmp_path, reload_cc):
    box = tmp_path / "box.env"
    box.write_text("ASVPROJECT_THERMAL_ROTATE_DEG=180\n")
    cc = reload_cc(ASVPROJECT_BOX_CONFIG=str(box), ASVPROJECT_THERMAL_ROTATE_DEG="0")
    assert cc.THERMAL_ROTATE_DEG == 0


def test_box_env_missing_file_is_not_an_error(tmp_path, reload_cc):
    cc = reload_cc(ASVPROJECT_BOX_CONFIG=str(tmp_path / "nope.env"))
    assert cc.THERMAL_ROTATE_DEG == 0


# --- radar profile provenance ---------------------------------------------
# On 2026-08-19 both halves of the clutterRemoval A/B silently ran the default
# profile: ASVPROJECT_RADAR_CONFIG never reached the radar, nothing on disk
# recorded which profile a clip used, and the null result was only diagnosed
# days later. These cover the provenance that makes that detectable.

CFG_OFF = """% a comment line the CLI must never see
sensorStop
channelCfg 15 7 0
cfarCfg -1 0 2 8 4 3 0 10 0
clutterRemoval -1 0
sensorStart
"""
CFG_ON = CFG_OFF.replace("clutterRemoval -1 0", "clutterRemoval -1 1")


def test_profile_summary_reports_the_ab_setting(tmp_path, reload_cc):
    cc = reload_cc()
    off = tmp_path / "off.cfg"; off.write_text(CFG_OFF)
    on = tmp_path / "on.cfg"; on.write_text(CFG_ON)
    s_off = cc.radar_profile_summary(str(off))
    s_on = cc.radar_profile_summary(str(on))
    assert "clutterRemoval -1 0" in s_off
    assert "clutterRemoval -1 1" in s_on
    # The whole point: the two halves of an A/B must not read alike.
    assert s_off != s_on


def test_profile_summary_skips_comments_and_keeps_key_lines(tmp_path, reload_cc):
    cc = reload_cc()
    p = tmp_path / "c.cfg"; p.write_text(CFG_OFF)
    s = cc.radar_profile_summary(str(p))
    assert "a comment line" not in s
    assert "channelCfg 15 7 0" in s
    assert "sensorStart" not in s          # not a profile key


def test_profile_summary_survives_a_missing_file(tmp_path, reload_cc):
    cc = reload_cc()
    assert "unreadable" in cc.radar_profile_summary(str(tmp_path / "nope.cfg"))


def test_write_radar_profile_records_source_and_body(tmp_path, reload_cc):
    cc = reload_cc()
    src = tmp_path / "on.cfg"; src.write_text(CFG_ON)
    out = tmp_path / "clip"; out.mkdir()
    cc.write_radar_profile(out, str(src))
    body = (out / "radar_profile.cfg").read_text()
    assert f"% source: {src}" in body
    assert "% summary: " in body
    assert "clutterRemoval -1 1" in body
    # The full config is kept, so a clip can be replayed exactly.
    assert "sensorStart" in body


def test_write_radar_profile_does_not_raise_on_a_bad_source(tmp_path, reload_cc):
    cc = reload_cc()
    out = tmp_path / "clip"; out.mkdir()
    cc.write_radar_profile(out, str(tmp_path / "nope.cfg"))
    assert "could not read" in (out / "radar_profile.cfg").read_text()


# === ThermalReader (latest-wins thermal drain, 2026-08-19 incident) ========

class _FakeThermalCap:
    """Scripted cv2-capture stand-in: yields queued frames then fails."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        import time
        time.sleep(0.01)   # emulate a stalled sensor read
        return False, None

    def release(self):
        self.released = True


def _wait_until(pred, timeout=2.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_thermal_reader_take_fresh_is_latest_wins_and_take_once():
    from scripts.data_collection.continuous_capture import ThermalReader
    frames = [np.full((4, 4, 3), i, np.uint8) for i in (1, 2, 3)]
    reader = ThermalReader(_FakeThermalCap(frames))
    reader.start()
    try:
        assert _wait_until(lambda: reader.reads_ok == 3)
        taken = reader.take_fresh()
        assert taken is not None and taken[0, 0, 0] == 3  # newest, not first
        assert reader.take_fresh() is None                # take-once contract
    finally:
        reader.release()


def test_thermal_reader_peek_does_not_consume():
    from scripts.data_collection.continuous_capture import ThermalReader
    reader = ThermalReader(_FakeThermalCap([np.zeros((4, 4, 3), np.uint8)]))
    reader.start()
    try:
        assert reader.peek(timeout=2.0) is not None
        assert reader.take_fresh() is not None  # peek left it unconsumed
    finally:
        reader.release()


def test_thermal_reader_dead_sensor_counts_failures_and_releases():
    from scripts.data_collection.continuous_capture import ThermalReader
    cap = _FakeThermalCap([])   # dead from the start
    reader = ThermalReader(cap)
    reader.start()
    assert _wait_until(lambda: reader.reads_fail > 0)
    assert reader.peek(timeout=0.05) is None
    assert reader.take_fresh() is None
    reader.release()
    assert cap.released
    assert not reader.is_alive()


# --- GPS sidecar (gps_<ts>.csv, since 2026-08-24) ---------------------------
#
# Own-boat position read from boat1's autopilot log. Metadata, not a sensor:
# every failure mode has to degrade to "no rows" and never cost a frame.

class _StubGPS:
    """Stands in for GPSReader: no threads, no ssh, a scripted fix sequence."""

    def __init__(self, fixes):
        self._fixes = list(fixes)
        self._taken = None
        self.closed = False

    def _current(self):
        return self._fixes[0] if self._fixes else None

    def advance(self):
        if len(self._fixes) > 1:
            self._fixes.pop(0)

    def get(self, live_only=False):
        return self._current()

    def take_new(self):
        fix = self._current()
        if fix is None:
            return None
        key = (fix.source, fix.lat_deg, fix.lon_deg)
        if key == self._taken:
            return None
        self._taken = key
        return fix

    def describe(self):
        return "stub"

    def close(self):
        self.closed = True


def _fix(lat=46.000000, lon=9.000000, source="boat_log", quality=4, when=None):
    from datetime import datetime, timezone

    from scripts.sensor_processing.gps_boat1 import GPSFix
    return GPSFix(lat, lon, fix_quality=quality, source=source,
                  fix_time_utc=when or datetime(2026, 8, 24, 10, 0,
                                                tzinfo=timezone.utc))


def _gps_rows(tmp_path):
    import csv as _csv

    path = sorted(tmp_path.glob("gps_*.csv"))[-1]
    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    return rows[0], rows[1:]


def test_gps_row_records_source_and_the_sources_own_time(reload_cc):
    """Date/Time is when WE wrote the row; FixTime is when the fix was taken.
    Both, because they differ by up to the ~30 min poll interval and only the
    pair makes a repeated row recognisable as the same reading."""
    from datetime import datetime

    cc = reload_cc()
    row = cc.gps_row(_fix(), datetime(2026, 8, 24, 12, 0, 0, 123456))
    assert row[0] == "2026-08-24" and row[1] == "12:00:00.1"
    assert row[2] == "46.0000000" and row[3] == "9.0000000"
    assert row[4] == 4 and row[5] == "boat_log"
    assert row[6] == "2026-08-24T10:00:00+00:00"


def test_gps_row_leaves_unknown_quality_empty_not_zero(reload_cc):
    """GGA 0 means 'no fix'; an empty cell means 'not reported'. Writing 0 for
    the second would mark a good RTK position as fixless."""
    from datetime import datetime

    cc = reload_cc()
    row = cc.gps_row(_fix(quality=None, source="fallback"), datetime.now())
    assert row[4] == "" and row[5] == "fallback"


def test_chunk_writes_the_position_in_force_at_chunk_start(
        reload_cc, monkeypatch, tmp_path):
    """Every clip is self-contained: it carries a position even when the ~30
    min poll did not come round during it."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="0")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(None, None, None, None, _StubGPS([_fix()]))

    header, rows = _gps_rows(tmp_path)
    assert header == list(cc.GPS_CSV_COLUMNS)
    assert len(rows) == 1 and rows[0][5] == "boat_log"


def test_chunk_writes_a_row_when_the_fix_changes_mid_chunk(
        reload_cc, monkeypatch, tmp_path):
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="1", ASVPROJECT_FPS="20")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    gps = _StubGPS([_fix(lat=47.1), _fix(lat=47.2)])
    original = gps.take_new
    calls = {"n": 0}

    def _take():
        calls["n"] += 1
        if calls["n"] == 3:
            gps.advance()          # a new poll lands partway through the chunk
        return original()

    gps.take_new = _take
    cc.write_one_chunk(None, None, None, None, gps)

    _header, rows = _gps_rows(tmp_path)
    assert [r[2] for r in rows] == ["47.1000000", "47.2000000"]


def test_chunk_does_not_repeat_an_unchanged_fix_every_loop(
        reload_cc, monkeypatch, tmp_path):
    """A 30 min cadence must not produce one row per capture loop."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="1", ASVPROJECT_FPS="20")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(None, None, None, None, _StubGPS([_fix()]))
    _header, rows = _gps_rows(tmp_path)
    assert len(rows) == 1


def test_chunk_with_no_position_writes_a_header_only_sidecar(
        reload_cc, monkeypatch, tmp_path):
    """The on-disk record of 'GPS was expected here and said nothing', matching
    the radar's header-only CSV convention."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="0")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(None, None, None, None, _StubGPS([]))
    _header, rows = _gps_rows(tmp_path)
    assert rows == []


def test_chunk_without_gps_writes_no_sidecar_at_all(
        reload_cc, monkeypatch, tmp_path):
    """Same rule as a disabled radar: no reader means no file, so the clip is
    an honest recording rather than a stream with a dead leg."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="0")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(None, None, None, None)
    assert not list(tmp_path.glob("gps_*.csv"))


def test_gps_read_failure_never_costs_a_frame(reload_cc, monkeypatch, tmp_path):
    """GPS is metadata: a reader that throws must not take the capture with it."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="1", ASVPROJECT_FPS="20")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)

    class _Exploding(_StubGPS):
        def take_new(self):
            raise RuntimeError("ssh went away")

    n_frames = cc.write_one_chunk(None, None, None, None, _Exploding([_fix()]))
    assert n_frames > 0
    assert list(tmp_path.glob("frames_*.csv"))


def test_gps_enable_flag_is_read_at_import(reload_cc):
    assert reload_cc(ASVPROJECT_GPS_ENABLE="0").GPS_ENABLE is False
    assert reload_cc().GPS_ENABLE is True


# === Per-boot session folders ==============================================

def test_session_capture_dir_is_base_slash_stamp(tmp_path):
    from scripts.data_collection import continuous_capture as cc
    assert (cc.session_capture_dir(tmp_path, "2026-08-25_09-30-00")
            == tmp_path / "2026-08-25_09-30-00")


def test_session_capture_dir_default_stamp_matches_chunk_format(tmp_path):
    """Unstamped calls (what main() does at boot) must name the folder in the
    same %Y-%m-%d_%H-%M-%S format as the chunk files it will contain."""
    import re

    from scripts.data_collection import continuous_capture as cc
    d = cc.session_capture_dir(tmp_path)
    assert d.parent == tmp_path
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", d.name)


def test_session_subdir_flag_is_read_at_import(reload_cc):
    assert reload_cc().SESSION_SUBDIR is True
    assert reload_cc(ASVPROJECT_SESSION_SUBDIR="0").SESSION_SUBDIR is False


def test_write_one_chunk_ignores_session_subdir(reload_cc, monkeypatch, tmp_path):
    """The subdir is main()-only: smoke tools drive write_one_chunk directly
    and must keep writing flat into their own out_dir."""
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="1", ASVPROJECT_FPS="20")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    assert cc.SESSION_SUBDIR is True
    cc.write_one_chunk(None, None, None, None, None)
    assert list(tmp_path.glob("frames_*.csv"))  # flat, no subfolder
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]


# === Per-frame exposure metadata in the frames sidecar (2026-08-28) ========

class _StubRequest:
    """picamera2 CompletedRequest look-alike: frame + ISP metadata."""

    def __init__(self, meta):
        self._meta = meta

    def make_array(self, name):
        from scripts.data_collection import continuous_capture as cc
        w, h = cc.FISHEYE_SIZE
        return np.full((h, w, 3), 90, dtype=np.uint8)

    def get_metadata(self):
        return dict(self._meta)

    def release(self):
        pass


class _StubFish:
    def __init__(self, meta):
        self.meta = meta

    def capture_request(self):
        return _StubRequest(self.meta)


class _LegacyFish:
    """capture_array only (no request API): columns must come out blank."""

    def capture_array(self):
        from scripts.data_collection import continuous_capture as cc
        w, h = cc.FISHEYE_SIZE
        return np.full((h, w, 3), 90, dtype=np.uint8)


def test_frames_sidecar_carries_exposure_metadata(reload_cc, monkeypatch, tmp_path):
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="1", ASVPROJECT_FPS="20")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    fish = _StubFish({"ExposureTime": 299_998, "AnalogueGain": 7.9994,
                      "DigitalGain": 1.0, "Lux": 3.21, "ColourGains": (1.8, 1.6)})
    n = cc.write_one_chunk(fish, None, None, None, None)
    assert n > 0
    df = pd.read_csv(next(tmp_path.glob("frames_*.csv")))
    assert list(df.columns) == ["frame_index", "Date", "Time",
                                "ExposureTime", "AnalogueGain", "DigitalGain", "Lux"]
    assert df["ExposureTime"].iloc[0] == 299_998          # int microseconds
    assert df["AnalogueGain"].iloc[0] == pytest.approx(7.9994)
    assert df["Lux"].iloc[0] == pytest.approx(3.21)
    assert "ColourGains" not in df.columns                 # only the four we keep


def test_frames_sidecar_blank_exposure_without_request_api(reload_cc, monkeypatch, tmp_path):
    cc = reload_cc(ASVPROJECT_CHUNK_SECONDS="1", ASVPROJECT_FPS="20")
    monkeypatch.setattr(cc, "CAPTURE_DIR", tmp_path)
    cc.write_one_chunk(_LegacyFish(), None, None, None, None)
    df = pd.read_csv(next(tmp_path.glob("frames_*.csv")))
    assert list(df.columns)[:3] == ["frame_index", "Date", "Time"]
    assert df["ExposureTime"].isna().all() and df["Lux"].isna().all()


def test_exposure_cols_formatting():
    from scripts.data_collection import continuous_capture as cc
    assert cc._exposure_cols({}) == ["", "", "", ""]
    assert cc._exposure_cols({"ExposureTime": 33333.0, "AnalogueGain": 1.23456789}) \
        == [33333, 1.2346, "", ""]
