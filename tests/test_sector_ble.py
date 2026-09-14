"""BLE sector link: codec round-trip, framing, tail, loopback end-to-end.

Laptop-only — no bluetooth anywhere: the real transport (bluezero) is lazily
imported by sector_ble_tx.BluezeroTransport and never touched here. The wire
format contract lives in scripts/data_collection/sector_codec.py (vendored
by boatv1-boat-b's sector_rx.py; these tests are the guard on both ends).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from scripts.data_collection.sector_codec import (
    MAGIC,
    SAFE_NOTIFY_BYTES,
    VERSION,
    CodecError,
    Reassembler,
    decode_packet,
    deciseconds_to_ts,
    encode_record,
    fragment,
    ts_to_deciseconds,
)
from scripts.data_collection.sector_ble_tx import (
    JsonlTail,
    LoopbackTransport,
    main,
    resolve_shadow_jsonl,
    run_tx,
)


def _record(**over) -> dict:
    rec = {
        "protocol": 0,
        "timestamp": "16:21:08.3",
        "clip_id": "live/session",
        "bin_centers_deg": [-45, -30, -15, 0, 15, 30, 45],
        "scores": [0.0, 0.33, 0.66, 1.0, 0.66, 0.33, 0.0],
        "min_range_m": [None, 3.42, 2.1, 1.05, 8.97, None, None],
        "sensor_hits": [[0, 0, 0]] * 7,
        "tracked": False,
        "p_obstacle": [0.01, 0.27, 0.87, 0.99, 0.5, 0.29, 0.0],
        "shadow": {"tick_ms": 132.4, "seg_age_s": 0.5, "seg_fresh": True,
                   "scorer_ok": True, "source_latency_s": 1.23},
    }
    rec.update(over)
    return rec


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------

class TestCodec:
    def test_roundtrip_within_tolerance(self):
        rec = _record()
        out = decode_packet(encode_record(rec, seq=300))
        assert out["seq"] == 300
        assert out["timestamp"] == rec["timestamp"]
        assert out["bin_centers_deg"] == rec["bin_centers_deg"]
        for a, b in zip(out["scores"], rec["scores"]):
            assert abs(a - b) <= 0.0025
        for a, b in zip(out["p_obstacle"], rec["p_obstacle"]):
            assert abs(a - b) <= 0.0025
        for a, b in zip(out["min_range_m"], rec["min_range_m"]):
            if b is None:
                assert a is None
            else:
                assert a == pytest.approx(b, abs=0.005)
        sh = out["shadow"]
        assert sh["scorer_ok"] is True
        assert sh["seg_fresh"] is True
        assert sh["tick_ms"] == 132
        assert sh["source_latency_s"] == pytest.approx(1.23, abs=0.005)

    def test_seven_bin_packet_size(self):
        assert len(encode_record(_record(), 0)) == 46
        rec = _record()
        rec.pop("p_obstacle")
        assert len(encode_record(rec, 0)) == 39

    def test_optional_fields_absent(self):
        rec = _record()
        rec.pop("p_obstacle")
        rec["shadow"] = {"tick_ms": 10, "scorer_ok": False, "seg_fresh": False}
        out = decode_packet(encode_record(rec, 1))
        assert "p_obstacle" not in out
        assert out["shadow"]["source_latency_s"] is None
        assert out["shadow"]["scorer_ok"] is False

    def test_no_shadow_block_at_all(self):
        rec = _record()
        rec.pop("shadow")
        out = decode_packet(encode_record(rec, 1))
        assert out["shadow"]["tick_ms"] == 0
        assert out["shadow"]["scorer_ok"] is False

    def test_seq_wraps_uint16(self):
        assert decode_packet(encode_record(_record(), 65536 + 7))["seq"] == 7

    def test_null_and_clamped_ranges(self):
        rec = _record(min_range_m=[None, 0.0, 700.0, 1.0, None, None, None])
        out = decode_packet(encode_record(rec, 0))
        assert out["min_range_m"][0] is None
        assert out["min_range_m"][1] == 0.0
        assert out["min_range_m"][2] == pytest.approx(655.34)

    def test_timestamp_helpers(self):
        assert deciseconds_to_ts(ts_to_deciseconds("00:00:00.0")) == "00:00:00.0"
        assert deciseconds_to_ts(ts_to_deciseconds("23:59:59.9")) == "23:59:59.9"

    def test_bad_magic_rejected(self):
        pkt = bytearray(encode_record(_record(), 0))
        pkt[0] = ord("X")
        with pytest.raises(CodecError, match="magic"):
            decode_packet(bytes(pkt))

    def test_bad_version_rejected(self):
        pkt = bytearray(encode_record(_record(), 0))
        pkt[2] = VERSION + 1
        # keep the checksum valid so it is the VERSION check that fires
        pkt[-1] = (pkt[-1] - 1) & 0xFF
        with pytest.raises(CodecError, match="version"):
            decode_packet(bytes(pkt))

    def test_corruption_rejected_by_checksum(self):
        pkt = bytearray(encode_record(_record(), 0))
        pkt[20] ^= 0x40
        with pytest.raises(CodecError, match="checksum"):
            decode_packet(bytes(pkt))

    def test_truncation_rejected(self):
        pkt = encode_record(_record(), 0)
        with pytest.raises(CodecError):
            decode_packet(pkt[:-3])
        with pytest.raises(CodecError):
            decode_packet(b"")

    def test_unencodable_records_raise(self):
        with pytest.raises(CodecError):
            encode_record({"timestamp": "16:00:00.0"}, 0)     # no bins
        with pytest.raises(CodecError):
            encode_record(_record(bin_centers_deg=[-45, -30, -10, 0, 15, 30, 45]), 0)
        with pytest.raises(CodecError):
            encode_record(_record(scores=[1.0]), 0)           # length mismatch
        with pytest.raises(CodecError):
            encode_record([], 0)                              # not a dict

    def test_magic_constant(self):
        assert encode_record(_record(), 0)[:2] == MAGIC


# ---------------------------------------------------------------------------
# fragmentation / reassembly
# ---------------------------------------------------------------------------

class TestFraming:
    def test_single_notification_when_mtu_allows(self):
        pkt = encode_record(_record(), 5)
        frags = fragment(pkt, 5, notify_bytes=244)
        assert len(frags) == 1
        assert Reassembler().feed(frags[0]) == pkt

    def test_mtu_floor_multi_fragment(self):
        pkt = encode_record(_record(), 5)
        frags = fragment(pkt, 5, notify_bytes=SAFE_NOTIFY_BYTES)
        assert len(frags) == 3
        assert all(len(f) <= SAFE_NOTIFY_BYTES for f in frags)
        r = Reassembler()
        outs = [r.feed(f) for f in frags]
        assert outs[:-1] == [None, None]
        assert outs[-1] == pkt

    def test_lost_fragment_drops_partial_next_packet_survives(self):
        p1 = encode_record(_record(), 1)
        p2 = encode_record(_record(timestamp="16:21:09.3"), 2)
        f1 = fragment(p1, 1, 20)
        f2 = fragment(p2, 2, 20)
        r = Reassembler()
        r.feed(f1[0])                      # fragment 1 of p1 lost
        assert r.feed(f1[2]) is None       # discontinuity -> dropped
        outs = [r.feed(f) for f in f2]
        assert outs[-1] == p2
        assert r.dropped_partials == 1
        assert decode_packet(outs[-1])["timestamp"] == "16:21:09.3"

    def test_reconnect_mid_packet(self):
        p1 = encode_record(_record(), 1)
        p2 = encode_record(_record(), 2)
        r = Reassembler()
        r.feed(fragment(p1, 1, 20)[0])     # then the link drops
        outs = [r.feed(f) for f in fragment(p2, 2, 20)]
        assert outs[-1] == p2


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------

class TestTail:
    def test_follows_growing_file_partial_lines_held(self, tmp_path):
        path = tmp_path / "shadow_x.jsonl"
        tail = JsonlTail(path)
        assert tail.read_new_lines() == []          # not created yet
        with open(path, "w") as fh:
            fh.write('{"a": 1}\n{"b": ')
            fh.flush()
            assert tail.read_new_lines() == ['{"a": 1}']
            fh.write('2}\n')
            fh.flush()
            assert tail.read_new_lines() == ['{"b": 2}']
        assert tail.read_new_lines() == []

    def test_from_end_skips_history(self, tmp_path):
        path = tmp_path / "shadow_x.jsonl"
        path.write_text('{"old": 1}\n')
        tail = JsonlTail(path, from_end=True)
        assert tail.read_new_lines() == []
        with open(path, "a") as fh:
            fh.write('{"new": 2}\n')
        assert tail.read_new_lines() == ['{"new": 2}']

    def test_resolve_newest_shadow_incl_subdir(self, tmp_path):
        assert resolve_shadow_jsonl(tmp_path) is None
        (tmp_path / "shadow_2026-09-02_10-00-00.jsonl").touch()
        sub = tmp_path / "2026-09-03_08-00-00"
        sub.mkdir()
        newest = sub / "shadow_2026-09-03_08-00-01.jsonl"
        newest.touch()
        assert resolve_shadow_jsonl(tmp_path) == newest


# ---------------------------------------------------------------------------
# loopback end-to-end
# ---------------------------------------------------------------------------

class TestLoopback:
    def _write(self, path: Path, n=3, malformed=True):
        with open(path, "w") as fh:
            for i in range(n):
                fh.write(json.dumps(_record(timestamp=f"16:21:0{i}.0")) + "\n")
            if malformed:
                fh.write("not json at all {{{\n")
                fh.write(json.dumps({"timestamp": "16:21:09.0"}) + "\n")  # no bins

    def test_run_tx_end_to_end(self, tmp_path):
        path = tmp_path / "shadow_a.jsonl"
        self._write(path)
        transport = LoopbackTransport(verbose=False)
        stats = run_tx(JsonlTail(path), transport, idle_timeout_s=0.3)
        assert stats.sent == 3
        assert stats.skipped == 2
        assert stats.notifications == 9        # 46 B / 18 B payload = 3 frags
        assert [r["timestamp"] for r in transport.records] == [
            "16:21:00.0", "16:21:01.0", "16:21:02.0"]
        assert transport.decode_errors == 0
        assert transport.records[0]["seq"] == 0
        assert transport.records[2]["p_obstacle"][3] == pytest.approx(0.99, abs=0.0025)

    def test_run_tx_picks_up_lines_written_while_running(self, tmp_path):
        path = tmp_path / "shadow_a.jsonl"
        path.write_text(json.dumps(_record()) + "\n")
        transport = LoopbackTransport(verbose=False)

        def writer():
            time.sleep(0.15)
            with open(path, "a") as fh:
                fh.write(json.dumps(_record(timestamp="17:00:00.0")) + "\n")

        t = threading.Thread(target=writer)
        t.start()
        stats = run_tx(JsonlTail(path), transport, idle_timeout_s=0.6)
        t.join()
        assert stats.sent == 2
        assert transport.records[-1]["timestamp"] == "17:00:00.0"

    def test_session_dir_roll_to_newer_file(self, tmp_path):
        old = tmp_path / "shadow_2026-09-03_10-00-00.jsonl"
        old.write_text(json.dumps(_record()) + "\n")
        transport = LoopbackTransport(verbose=False)

        def writer():
            time.sleep(0.3)
            new = tmp_path / "shadow_2026-09-03_10-05-00.jsonl"
            with open(new, "w") as fh:
                fh.write(json.dumps(_record(timestamp="18:00:00.0")) + "\n")

        t = threading.Thread(target=writer)
        t.start()
        import scripts.data_collection.sector_ble_tx as tx
        orig = tx.RESOLVE_S
        tx.RESOLVE_S = 0.05
        try:
            stats = run_tx(JsonlTail(old), transport, session_dir=tmp_path,
                           idle_timeout_s=1.0)
        finally:
            tx.RESOLVE_S = orig
        t.join()
        assert stats.sent == 2
        assert transport.records[-1]["timestamp"] == "18:00:00.0"

    def test_cli_loopback(self, tmp_path, capsys):
        path = tmp_path / "shadow_a.jsonl"
        self._write(path, n=2)
        rc = main(["--tail", str(path), "--loopback", "--idle-timeout", "0.3"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 records decoded, 0 decode errors" in out

    def test_cli_loopback_bigger_notify(self, tmp_path):
        path = tmp_path / "shadow_a.jsonl"
        self._write(path, n=2, malformed=False)
        transport = LoopbackTransport(verbose=False)
        stats = run_tx(JsonlTail(path), transport, notify_bytes=180,
                       idle_timeout_s=0.3)
        assert stats.sent == 2
        assert stats.notifications == 2        # one notification per record


def test_yolo_fresh_flag_roundtrips_and_is_backward_compatible():
    from scripts.data_collection import sector_codec as c
    rec = {"timestamp": "12:00:00.5", "bin_centers_deg": [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0],
           "scores": [0.0] * 7, "min_range_m": [None] * 7, "p_obstacle": [0.3] * 7,
           "shadow": {"tick_ms": 1700, "scorer_ok": True, "seg_fresh": False, "yolo_fresh": True}}
    d = c.decode_packet(c.encode_record(rec, 7))
    assert d["shadow"]["yolo_fresh"] is True and d["shadow"]["seg_fresh"] is False
    rec["shadow"].pop("yolo_fresh")
    assert c.decode_packet(c.encode_record(rec, 8))["shadow"]["yolo_fresh"] is False
    assert len(c.encode_record(rec, 9)) == 46          # size unchanged
