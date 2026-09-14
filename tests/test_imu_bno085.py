import math
import struct
import time

import numpy as np
import pytest

from scripts.sensor_processing.imu_bno085 import (
    Attitude,
    BNO085Reader,
    decode_rvc_frame,
    load_imu_config,
    parse_packet,
    parse_rvc_frame,
    quaternion_to_euler,
)
from scripts.utils.geometry import UP_LEVEL


def _rvc_frame(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, ax=0, ay=0, az=0, index=0):
    """Build a valid 19-byte UART-RVC frame (checksum computed)."""
    body = struct.pack(
        "<BhhhhhhBBB", index & 0xFF,
        round(yaw_deg * 100), round(pitch_deg * 100), round(roll_deg * 100),
        ax, ay, az, 0, 0, 0)          # 16 bytes = bytes[2:18]
    return b"\xaa\xaa" + body + bytes([sum(body) & 0xFF])


# --- quaternion -> euler ----------------------------------------------------

def test_quat_identity_is_zero():
    roll, pitch, yaw = quaternion_to_euler(1.0, 0.0, 0.0, 0.0)
    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 0.0))


def test_quat_90deg_roll():
    # rotation of 90 deg about x: w=cos(45), x=sin(45)
    c = math.cos(math.pi / 4)
    roll, pitch, yaw = quaternion_to_euler(c, c, 0.0, 0.0)
    assert roll == pytest.approx(math.pi / 2, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)


def test_quat_unnormalised_is_handled():
    # doubling the quaternion must not change the angles
    roll, pitch, yaw = quaternion_to_euler(2.0, 0.0, 0.0, 0.0)
    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 0.0))


def test_quat_zero_is_safe():
    assert quaternion_to_euler(0.0, 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_quat_pitch_clamps_at_pole():
    # gimbal lock: sinp >= 1 -> pitch = +90 deg, no domain error
    _r, pitch, _y = quaternion_to_euler(c := math.cos(math.pi / 4), 0.0, c, 0.0)
    assert abs(pitch) == pytest.approx(math.pi / 2, abs=1e-6)


# --- packet parsing ---------------------------------------------------------

def test_parse_quat_f32_le_roundtrip():
    data = struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0)
    att = parse_packet(data, "quat_f32_le")
    assert att is not None
    assert att.pitch_rad == pytest.approx(0.0)
    assert att.roll_rad == pytest.approx(0.0)


def test_parse_euler_deg():
    data = struct.pack("<fff", 10.0, -5.0, 90.0)  # roll, pitch, yaw degrees
    att = parse_packet(data, "euler_f32_le_deg")
    assert att.roll_deg == pytest.approx(10.0, abs=1e-4)
    assert att.pitch_deg == pytest.approx(-5.0, abs=1e-4)
    assert att.yaw_deg == pytest.approx(90.0, abs=1e-4)


def test_parse_csv():
    att = parse_packet(b"3.0,-2.0,45.0\n", "csv")
    assert att.roll_deg == pytest.approx(3.0)
    assert att.pitch_deg == pytest.approx(-2.0)


def test_parse_short_packet_returns_none():
    assert parse_packet(b"\x00\x00", "quat_f32_le") is None
    assert parse_packet(b"x", "csv") is None


def test_parse_unknown_fmt_raises():
    with pytest.raises(ValueError):
        parse_packet(struct.pack("<ffff", 1, 0, 0, 0), "bogus")


# --- UART-RVC frame decode --------------------------------------------------

def test_decode_rvc_frame_roundtrip():
    d = decode_rvc_frame(_rvc_frame(yaw_deg=12.34, pitch_deg=-5.6, roll_deg=7.8,
                                    ax=100, ay=-50, az=1000))
    assert d is not None
    yaw, pitch, roll, axm, aym, azm = d
    assert yaw == pytest.approx(12.34, abs=0.01)
    assert pitch == pytest.approx(-5.6, abs=0.01)
    assert roll == pytest.approx(7.8, abs=0.01)
    assert axm == pytest.approx(100 * 9.80665 / 1000, abs=1e-4)
    assert azm == pytest.approx(1000 * 9.80665 / 1000, abs=1e-4)


def test_parse_rvc_frame_to_attitude():
    att = parse_rvc_frame(_rvc_frame(yaw_deg=90.0, pitch_deg=10.0, roll_deg=-20.0))
    assert att is not None
    assert att.yaw_deg == pytest.approx(90.0, abs=0.01)
    assert att.pitch_deg == pytest.approx(10.0, abs=0.01)
    assert att.roll_deg == pytest.approx(-20.0, abs=0.01)
    assert att.source == "bno085:uart_rvc"


def test_decode_rvc_bad_checksum_returns_none():
    f = bytearray(_rvc_frame(yaw_deg=1.0))
    f[-1] ^= 0xFF
    assert decode_rvc_frame(bytes(f)) is None


def test_decode_rvc_bad_header_returns_none():
    f = bytearray(_rvc_frame())
    f[0] = 0x00
    assert decode_rvc_frame(bytes(f)) is None


def test_decode_rvc_short_returns_none():
    assert decode_rvc_frame(b"\xaa\xaa\x00") is None


# --- Attitude helpers -------------------------------------------------------

def test_attitude_up_vector_level():
    np.testing.assert_allclose(Attitude(0.0, 0.0).up_vector_camera(), UP_LEVEL, atol=1e-12)


def test_attitude_degrees_properties():
    att = Attitude(pitch_rad=math.radians(12.0), roll_rad=math.radians(-3.0))
    assert att.pitch_deg == pytest.approx(12.0)
    assert att.roll_deg == pytest.approx(-3.0)


# --- config -----------------------------------------------------------------

def test_load_imu_config_missing_returns_defaults(tmp_path):
    cfg = load_imu_config(tmp_path / "nope.yaml")
    assert cfg["backend"] == "simulation"
    assert cfg["enabled"] is False


def test_load_imu_config_real_file_loads():
    cfg = load_imu_config()  # the repo configs/imu.yaml
    assert "backend" in cfg and "packet_fmt" in cfg


# --- simulation reader (no hardware) ----------------------------------------

def test_simulation_reader_produces_bounded_attitude():
    cfg = load_imu_config()
    cfg.update({"backend": "simulation", "simulation_hz": 200.0,
                "simulation_roll_deg": 5.0, "simulation_pitch_deg": 2.0})
    with BNO085Reader(cfg) as reader:
        att = None
        for _ in range(50):
            att = reader.get()
            if att is not None:
                break
            time.sleep(0.01)
        assert att is not None
        assert reader.connected
        assert abs(att.roll_deg) <= 5.0 + 1e-6
        assert abs(att.pitch_deg) <= 2.0 + 1e-6
        assert att.source.startswith("bno085:")


def test_reader_get_none_when_stale():
    cfg = load_imu_config()
    cfg.update({"backend": "simulation", "stale_after_s": 0.0})
    reader = BNO085Reader(cfg)
    # never started -> no sample
    assert reader.get() is None


def test_mount_offset_applied():
    cfg = load_imu_config()
    cfg.update({"backend": "simulation", "simulation_roll_deg": 0.0,
                "simulation_pitch_deg": 0.0,
                "mount_offset_rad": {"pitch": 0.1, "roll": -0.05}})
    with BNO085Reader(cfg) as reader:
        att = None
        for _ in range(50):
            att = reader.get()
            if att is not None:
                break
            time.sleep(0.01)
        assert att is not None
        assert att.pitch_rad == pytest.approx(0.1, abs=1e-6)
        assert att.roll_rad == pytest.approx(-0.05, abs=1e-6)


# --- rotation-composition attitude source (live/replay parity) --------------

def test_publish_rotation_source_composes_camera_frame():
    """attitude_source: rotation makes the LIVE reader publish the same
    camera-frame pitch/roll/heading the replay provider derives, via the
    shared composition functions in imu_replay (parity by construction)."""
    from scripts.sensor_processing.imu_replay import (
        camera_heading_from_rvc,
        camera_pitch_roll_from_rvc,
    )
    cfg = load_imu_config()
    cfg.update({"backend": "uart_rvc", "attitude_source": "rotation",
                "mount_offset_rad": {"pitch": 0.0, "roll": 0.0}})
    reader = BNO085Reader(cfg)
    yaw, pitch, roll = math.radians(25.0), math.radians(77.9), math.radians(103.2)
    reader._publish(Attitude(pitch_rad=pitch, roll_rad=roll, yaw_rad=yaw))
    att = reader.get()
    want_p, want_r = camera_pitch_roll_from_rvc(pitch, roll)
    want_y = camera_heading_from_rvc(yaw, pitch, roll)
    assert att.pitch_rad == pytest.approx(float(want_p), abs=1e-9)
    assert att.roll_rad == pytest.approx(float(want_r), abs=1e-9)
    assert att.yaw_rad == pytest.approx(float(want_y), abs=1e-9)


def test_publish_simulation_backend_never_composes():
    """The simulation backend synthesizes camera-frame attitude directly, so
    rotation composition (which expects raw box-mount angles) must not
    apply to it even when the config asks for it."""
    cfg = load_imu_config()
    cfg.update({"backend": "simulation", "attitude_source": "rotation",
                "mount_offset_rad": {"pitch": 0.0, "roll": 0.0}})
    reader = BNO085Reader(cfg)
    reader._publish(Attitude(pitch_rad=0.02, roll_rad=-0.05, yaw_rad=0.5))
    att = reader.get()
    assert att.pitch_rad == pytest.approx(0.02)
    assert att.roll_rad == pytest.approx(-0.05)
    assert att.yaw_rad == pytest.approx(0.5)
