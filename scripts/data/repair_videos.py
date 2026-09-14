"""Repair capture videos in place-of-original: moov-less mp4s and the thermal
stuck row, written as `recovered/<kind>_<ts>.mp4` copies.

Two faults this box produces, both fixable after the fact:

1. **Hard power cut mid-chunk** leaves an mp4 without its `moov` atom (the
   index is written at close). The `mdat` payload is a raw MPEG-4 Part 2
   stream whose decoder configuration (the VOL header) lives only in the
   missing `moov/.../esds` — so decoders report "0x0". Borrowing the esds
   DecoderSpecificInfo from any healthy chunk of the same stream (same
   writer, same size) and prepending it makes the raw stream decodable;
   it is then re-encoded into a normal mp4.
2. **Stuck-hot Lepton FPA row** (sensor row 21 — row 98 in the 180-degree
   rotated capture frame; docs/guides/box_reassembly.md). The processing
   pipeline repairs it live (detection.yaml thermal.dead_rows), but a
   repaired copy on disk makes every consumer — dashboard bake included —
   see the clean frame. The row is DETECTED per file (per-row mean over the
   first frames, full-width test), so unrotated and rotated captures both
   get the right row.

Originals are never modified. `scripts.utils.datasets.resolve_triplet`
prefers `recovered/<name>_trimmed.mp4` > `recovered/<name>.mp4` > the
original, so the copies take effect everywhere without further wiring.
A chunk whose radar CSV is also empty still bakes (iterate_triplet
synthesizes the timeline from the video, as for the day-1 radar-dead clips)
but carries no radar layer — the tool says so.

    python -m scripts.data.repair_videos data/captures/2026-08-26_afloat
    python -m scripts.data.repair_videos data/captures/2026-08-19_afloat --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.utils.cv_common import repair_dead_rows

# ---------------------------------------------------------------------------
# mp4 box parsing (just enough to find moov and the esds decoder config)
# ---------------------------------------------------------------------------

_CONTAINER_SKIP = {"stsd": 8, "mp4v": 78, "esds": 4}


def _boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield typ.decode("latin1"), pos + hdr, pos + size
        pos += size


def find_box(buf: bytes, path: list[str], start: int = 0,
             end: int | None = None) -> tuple[int, int] | None:
    """(payload_start, box_end) of the box at `path`, or None."""
    end = len(buf) if end is None else end
    for typ, a, b in _boxes(buf, start, end):
        if typ == path[0]:
            if len(path) == 1:
                return a, b
            r = find_box(buf, path[1:], a + _CONTAINER_SKIP.get(typ, 0), b)
            if r:
                return r
    return None


def has_moov(path: Path) -> bool:
    return find_box(path.read_bytes(), ["moov"]) is not None


def mdat_payload_offset(path: Path) -> int | None:
    head = path.read_bytes()[:64] if path.stat().st_size >= 64 else path.read_bytes()
    for typ, a, _b in _boxes(head, 0, len(head)):
        if typ == "mdat":
            return a
    return None


def vol_header(healthy: Path) -> bytes:
    """MPEG-4 Part 2 decoder config (VOS/VO/VOL start codes) from a healthy
    file's esds DecoderSpecificInfo."""
    buf = healthy.read_bytes()
    r = find_box(buf, ["moov", "trak", "mdia", "minf", "stbl", "stsd", "mp4v", "esds"])
    if not r:
        raise ValueError(f"no mp4v esds in {healthy}")
    es = buf[r[0]:r[1]]
    i = es.find(b"\x00\x00\x01\xb0")
    if i < 0:
        i = es.find(b"\x00\x00\x01\x20")
    if i < 0:
        raise ValueError(f"no VOL start code in {healthy}")
    j = es.find(b"\x06\x01\x02", i)  # SLConfigDescriptor follows the DSI
    return es[i:j if j > 0 else None]


# ---------------------------------------------------------------------------
# video helpers
# ---------------------------------------------------------------------------

def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def hot_row(path: Path, nframes: int = 100, min_excess: float = 60.0,
            min_width: float = 0.6) -> tuple[int | None, float]:
    """(row, excess): the full-width stuck row, or (None, best excess)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(nframes):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 10:
        return None, 0.0
    a = np.stack(frames)
    rowmean = a.mean(axis=(0, 2))
    dev = rowmean - np.median(rowmean)
    r = int(np.argmax(dev))
    h = a.shape[1]
    nb = (a[:, max(r - 2, 0), :] + a[:, min(r + 2, h - 1), :]) / 2
    width = float(((a[:, r, :] - nb).mean(axis=0) > 40).mean())
    ok = dev[r] > min_excess and width > min_width
    return (r if ok else None), float(dev[r])


def _encode(frames, out: Path, w: int, h: int, fps: int = 3) -> tuple[int, int]:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "mpeg4", "-q:v", "1", "-r", str(fps), str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        p.stdin.write(f.tobytes())
        n += 1
    p.stdin.close()
    return n, p.wait()


def recover_moov(broken: Path, healthy: Path, out: Path, fps: int = 3) -> int:
    """Rebuild a decodable mp4 from a moov-less file. Returns the frame count
    of the result (0 = nothing recoverable)."""
    off = mdat_payload_offset(broken)
    if off is None:
        return 0
    raw = broken.read_bytes()[off:]
    with tempfile.TemporaryDirectory() as td:
        m4v = Path(td) / "raw.m4v"
        m4v.write_bytes(vol_header(healthy) + raw)
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "m4v",
                            "-framerate", str(fps), "-i", str(m4v),
                            "-c:v", "mpeg4", "-q:v", "1", "-r", str(fps), str(out)],
                           capture_output=True)
    if r.returncode != 0 or not out.exists():
        return 0
    n = frame_count(out)
    if n == 0:
        out.unlink(missing_ok=True)
    return n


def repair_rows_video(src: Path, out: Path, row: int, fps: int = 3) -> tuple[int, bool]:
    cap = cv2.VideoCapture(str(src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def gen():
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield repair_dead_rows(f, [row])

    n, rc = _encode(gen(), out, w, h, fps)
    cap.release()
    return n, rc == 0 and frame_count(out) == n


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def repair_mission(root: Path, dry_run: bool = False, kinds=("fisheye", "thermal")) -> list[str]:
    log: list[str] = []
    vids = sorted(Path(p) for p in glob.glob(f"{root}/**/*_20*.mp4", recursive=True))
    vids = [v for v in vids if "recovered" not in v.parts and "_smoketest" not in str(v)]
    for v in vids:
        kind = v.name.split("_")[0]
        if kind not in kinds:
            continue
        rel = v.relative_to(root)
        if v.stat().st_size == 0:
            log.append(f"EMPTY    {rel}")
            continue
        rec_dir = v.parent / "recovered"
        rec = rec_dir / v.name
        if rec.exists() or rec.with_name(rec.stem + "_trimmed.mp4").exists():
            log.append(f"SKIP     {rel} (recovered copy exists)")
            continue
        src = v
        tmp_src: Path | None = None
        if not has_moov(v):
            healthy = [h for h in sorted(v.parent.glob(f"{kind}_20*.mp4"))
                       if h != v and h.stat().st_size > 0 and has_moov(h)]
            if not healthy:
                log.append(f"NOREF    {rel} (no healthy sibling for the decoder header)")
                continue
            if dry_run:
                log.append(f"MOOV?    {rel}")
                continue
            rec_dir.mkdir(exist_ok=True)
            tmp_src = rec_dir / f".tmp_{v.name}"
            n = recover_moov(v, healthy[0], tmp_src)
            if n == 0:
                log.append(f"FAILREC  {rel}")
                tmp_src.unlink(missing_ok=True)
                continue
            radar = v.parent / v.name.replace(kind, "mmwave").replace(".mp4", ".csv")
            keyable = radar.exists() and radar.stat().st_size > 0
            log.append(f"MOOV     {rel} -> {n} frames"
                       + ("" if keyable else "  [radar CSV empty: frames keyed on a synthesized timeline, no radar layer]"))
            src = tmp_src
            if kind != "thermal":
                shutil.move(str(tmp_src), str(rec))
                continue
        if kind == "thermal":
            row, excess = hot_row(src)
            if row is None:
                if tmp_src is not None:
                    shutil.move(str(tmp_src), str(rec))
                    log.append(f"THERM    {rel} recovered; no stuck row (excess {excess:.0f})")
                else:
                    log.append(f"CLEAN    {rel} (excess {excess:.0f})")
                continue
            if dry_run:
                log.append(f"ROW{row:<4d}? {rel}")
                continue
            rec_dir.mkdir(exist_ok=True)
            n, ok = repair_rows_video(src, rec, row)
            if tmp_src is not None:
                tmp_src.unlink(missing_ok=True)
            log.append(f"ROW{row:<5d}{rel} -> recovered/{v.name} ({n} frames) {'OK' if ok else 'MISMATCH'}")
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="mission directories (recursive)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for root in args.roots:
        for line in repair_mission(Path(root), dry_run=args.dry_run):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
