"""One-command session ingest: Pi -> SSD archive -> validation -> bake ->
private R2 bucket -> dashboard catalogue.

    .venv/bin/python dashboard/tools/ingest_session.py \
        --pi vesselauser@<addr> --mission 2026-08-21_bench [--no-upload]

Steps (each printed, any failure stops before the next):
 1. PULL    rsync the Pi's ~/captures/ into the SSD mission dir, run rsync
            a second time, then md5-verify every file against the Pi
            (the 2026-08-19 dismount lesson: never trust one pass).
 2. VALIDATE per chunk: fisheye readable + frame count vs frames.csv span
            (loop-rate degradation like the 2026-08-19 thermal incident),
            thermal empty/undersized, radar rows + Doppler/SNR presence,
            IMU rows. Prints a table; corrupt chunks are called out but do
            not block (the baker skips them honestly).
 3. BAKE    bake_corpus.py --only <mission> (incremental: existing
            activities reuse; back-to-back chunks concatenate).
 4. UPLOAD  ingest_r2.py --prune (needs dashboard/.env.local).

The smoke-test dir (_smoketest) and radar_profile.cfg sidecars are pulled
but never baked (bake only walks fisheye_*.mp4 sets).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CAPTURES = REPO_ROOT / "data" / "captures"
DASHBOARD = REPO_ROOT / "dashboard"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def pull_and_verify(pi: str, mission_dir: Path, src: str, date: str) -> None:
    print(f"[1/4] PULL {pi}:{src} (files matching *{date}*) -> {mission_dir}")
    mission_dir.mkdir(parents=True, exist_ok=True)
    # Date-filtered: the Pi's capture dir accumulates already-archived
    # sessions; only this session's files belong in this mission dir.
    rsync = ["rsync", "-a", "--prune-empty-dirs",
             "--exclude", "_smoketest/",
             "--include", "*/", "--include", f"*{date}*",
             # per-boot session folders (captures/<stamp>/, 2026-08-25+) carry
             # radar_profile.cfg with no date in its name: take it whenever its
             # session dir matches the date (caught 2026-09-09: nine sessions
             # arrived without their profile record)
             "--include", f"*{date}*/radar_profile.cfg",
             "--exclude", "*",
             f"{pi}:{src}/", f"{mission_dir}/"]
    if run(rsync).returncode:
        raise SystemExit("rsync failed")
    if run(rsync).returncode:  # second pass: must be a no-op that succeeds
        raise SystemExit("rsync verify pass failed")
    # md5 the remote side and compare (BSD md5 locally, md5sum remotely)
    remote = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", pi,
         f"cd {src} && find . -path ./_smoketest -prune -o "
         f"-type f -name '*{date}*' -print0 | xargs -0 md5sum"],
        capture_output=True, text=True, check=True).stdout
    bad = 0
    n = 0
    for line in remote.strip().splitlines():
        digest, _, rel = line.partition("  ")
        local = mission_dir / rel.lstrip("./")
        if not local.exists():
            print(f"  MISSING locally: {rel}")
            bad += 1
            continue
        here = subprocess.run(["md5", "-q", str(local)], capture_output=True,
                              text=True, check=True).stdout.strip()
        n += 1
        if here != digest.strip():
            print(f"  CHECKSUM MISMATCH: {rel}")
            bad += 1
    if bad:
        raise SystemExit(f"verification FAILED for {bad} file(s) — nothing baked")
    print(f"  verified {n} files identical to the Pi")


def validate(mission_dir: Path) -> None:
    import cv2
    import pandas as pd

    print("[2/4] VALIDATE chunks")
    problems = 0
    chunks = sorted(mission_dir.rglob("fisheye_*.mp4"))
    for f in chunks:
        ts = f.stem.removeprefix("fisheye_")
        d = f.parent
        notes: list[str] = []
        cap = cv2.VideoCapture(str(f))
        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if vid_frames == 0:
            notes.append("FISHEYE UNREADABLE")
        frames_csv = d / f"frames_{ts}.csv"
        if frames_csv.exists():
            fr = pd.read_csv(frames_csv)
            if len(fr) > 1:
                t = pd.to_timedelta(fr["Time"])
                span = (t.iloc[-1] - t.iloc[0]).total_seconds()
                if span > 0 and len(fr) / span < 1.5:  # nominal 3 fps
                    notes.append(
                        f"LOOP-RATE DEGRADED ({len(fr)} frames over {span:.0f}s)")
        therm = d / f"thermal_{ts}.mp4"
        if therm.exists() and therm.stat().st_size < 10_000:
            notes.append(f"THERMAL EMPTY ({therm.stat().st_size} B)")
        mm = d / f"mmwave_{ts}.csv"
        radar_note = "no radar csv"
        if mm.exists():
            try:
                df = pd.read_csv(mm)
                pts = df.dropna(subset=["X"]) if "X" in df else df
                has_v = "V" in df.columns and pts["V"].notna().any()
                has_snr = "SNR" in df.columns and pts["SNR"].notna().any()
                radar_note = (f"radar {len(pts)} pts"
                              f"{' +doppler' if has_v else ' NO-DOPPLER'}"
                              f"{' +snr' if has_snr else ' NO-SNR'}")
                if len(pts) and not has_v:
                    notes.append("RADAR MISSING DOPPLER")
            except Exception as exc:
                notes.append(f"RADAR CSV UNREADABLE ({exc})")
        imu = d / f"imu_{ts}.csv"
        imu_note = "no imu csv"
        if imu.exists():
            imu_note = f"imu {max(sum(1 for _ in open(imu)) - 1, 0)} rows"
        status = "OK " if not notes else "WARN"
        problems += bool(notes)
        rel = d.relative_to(mission_dir)
        print(f"  [{status}] {rel}/{ts}: video {vid_frames}f · {radar_note} · "
              f"{imu_note}{(' · ' + '; '.join(notes)) if notes else ''}")
    print(f"  {len(chunks)} chunk(s), {problems} with warnings"
          f" (the baker skips unreadable ones honestly)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pi", default=None,
                    help="Pi ssh target (user@addr). Omit to skip the pull "
                         "and ingest an already-archived mission.")
    ap.add_argument("--src", default="~/captures",
                    help="capture dir on the Pi")
    ap.add_argument("--mission", required=True,
                    help="mission dir name under data/captures/ "
                         "(e.g. 2026-08-21_bench)")
    ap.add_argument("--date", default=None,
                    help="only pull files whose names contain this token "
                         "(default: the mission name's leading YYYY-MM-DD)")
    ap.add_argument("--no-upload", action="store_true",
                    help="stop after the bake (no R2/catalogue changes)")
    args = ap.parse_args()

    date = args.date or args.mission[:10]
    mission_dir = CAPTURES / args.mission
    if args.pi:
        pull_and_verify(args.pi, mission_dir, args.src, date)
    elif not mission_dir.exists():
        raise SystemExit(f"{mission_dir} does not exist and no --pi given")

    validate(mission_dir)

    print("[3/4] BAKE (concatenating back-to-back chunks) + ENRICH")
    py = str(REPO_ROOT / ".venv" / "bin" / "python")
    if run([py, str(DASHBOARD / "tools" / "bake_corpus.py"),
            "--only", args.mission]).returncode:
        raise SystemExit("bake failed")
    # environmental context for the data map + fusion features (GPS, sun,
    # luminance, weather, entities). Rebakes drop the block, so re-enrich
    # after every bake.
    if run([py, str(DASHBOARD / "tools" / "enrich_bundle.py")]).returncode:
        raise SystemExit("enrichment failed")

    if args.no_upload:
        print("[4/4] SKIPPED upload (--no-upload)")
        return 0
    print("[4/4] UPLOAD to R2 + catalogue (prune)")
    envfile = DASHBOARD / ".env.local"
    cmd = ["bash", "-c",
           f"set -a && source {envfile} && set +a && "
           f"npx tsx tools/ingest/ingest_r2.ts --prune"]
    if subprocess.run(cmd, cwd=DASHBOARD).returncode:
        raise SystemExit("upload failed")
    print("DONE — the new activities are live on the dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
