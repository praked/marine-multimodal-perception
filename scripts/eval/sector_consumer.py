"""Reference consumer for the sector-vector protocol (v0).

Reads the JSONL stream produced by `fusion.py --out` and prints the
blocked sectors per frame, plus the closest range across all bins
above an emergency-stop threshold. This is the simplest possible nav
client; the real navigation stack will run this kind of logic with
hysteresis, smoothing, and safety policy (see PLAN.md Branch D.2).

Usage:
    python -m scripts.eval.sector_consumer --input results/sectors_xxx.jsonl
    python -m scripts.eval.sector_consumer --input <path> --blocked 0.66 --hard-stop-m 2.0

Output is one line per frame:
    HH:MM:SS.f  blocked=[-30, -20]  closest=1.8m  HARD_STOP
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def decide(record: dict, blocked_threshold: float, hard_stop_m: float | None
           ) -> dict:
    """Apply a simple decision policy to one frame record.

    Returns {timestamp, blocked, closest, hard_stop, alert_bins,
    recommended_heading_deg, heading_reason, closest_ttc, fastest_closing}.
    Heading fields are None when the input is v0 or the recommender
    abstained (ambiguous / no recommendation).
    """
    bins = record["bin_centers_deg"]
    scores = record["scores"]
    ranges = record["min_range_m"]
    velocities = record.get("per_bin_velocity_mps") or [None] * len(bins)
    ttcs = record.get("per_bin_ttc_s") or [None] * len(bins)
    blocked: list[int] = []
    alert: list[int] = []
    for b, s in zip(bins, scores):
        if s >= blocked_threshold:
            blocked.append(int(b))
        elif s >= 0.33:
            alert.append(int(b))
    closest = None
    closest_blocked_bin = None
    for b, s, r in zip(bins, scores, ranges):
        if r is None or s < 0.33:
            continue
        if closest is None or r < closest:
            closest = r
            closest_blocked_bin = int(b)
    # Soonest collision: minimum non-null TTC across hit bins.
    closest_ttc = None
    closest_ttc_bin = None
    fastest_closing = None
    fastest_bin = None
    for b, s, v, t in zip(bins, scores, velocities, ttcs):
        if s < 0.33:
            continue
        if t is not None and (closest_ttc is None or t < closest_ttc):
            closest_ttc = t
            closest_ttc_bin = int(b)
        if v is not None and (fastest_closing is None or v > fastest_closing):
            fastest_closing = v
            fastest_bin = int(b)
    hard_stop = (
        hard_stop_m is not None
        and closest is not None
        and closest <= hard_stop_m
    )
    return {
        "timestamp": record["timestamp"],
        "blocked": blocked,
        "alert": alert,
        "closest": closest,
        "closest_bin": closest_blocked_bin,
        "closest_ttc": closest_ttc,
        "closest_ttc_bin": closest_ttc_bin,
        "fastest_closing": fastest_closing,
        "fastest_bin": fastest_bin,
        "hard_stop": hard_stop,
        "tracked": record.get("tracked", False),
        "protocol": record.get("protocol", 0),
        "recommended_heading_deg": record.get("recommended_heading_deg"),
        "heading_reason": record.get("heading_reason"),
        "smoothed_heading_deg": record.get("smoothed_heading_deg"),
        # v1.1 extensions (absent -> None): replayed/live IMU attitude,
        # per-bin seg free-space, per-bin radar<->camera confirmation.
        "attitude": record.get("attitude"),
        "free_space_m": record.get("free_space_m"),
        "confirmed": record.get("confirmed"),
        "_bins": bins,
    }


def _wrap_deg(a: float) -> float:
    """Wrap to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def absolute_heading(decision: dict, yaw_sign: float = 1.0) -> float | None:
    """Compass-frame steering heading = IMU yaw + (sign * bow-relative steer).

    The sector protocol's headings are BOW-RELATIVE (0 = straight ahead,
    positive = starboard). With the v1.1 `attitude` block present, adding the
    IMU yaw converts the smoothed steer into the IMU's absolute yaw frame so
    the autopilot can hand it to its own compass controller.

    `yaw_sign` handles the axis convention: the BNO08x ZYX yaw is
    right-hand-positive about up (counter-clockwise = port-positive), while
    our bearings are starboard-positive: VERIFY the sign on the boat against
    a known compass heading before trusting this in nav (one dock test:
    point the bow at a landmark, compare). Whether yaw is referenced to
    magnetic north or an arbitrary-but-stable power-on frame depends on which
    rotation-vector report boat1's bridge forwards, either works for
    differential steering; only absolute-north semantics change.
    """
    att = decision.get("attitude")
    steer = decision.get("smoothed_heading_deg")
    if att is None or steer is None or att.get("yaw_deg") is None:
        return None
    return _wrap_deg(float(att["yaw_deg"]) + yaw_sign * float(steer))


def format_line(decision: dict, yaw_sign: float = 1.0) -> str:
    parts = [decision["timestamp"]]
    if decision["blocked"]:
        parts.append(f"blocked={decision['blocked']}")
        conf = decision.get("confirmed")
        if conf:
            bins = decision.get("_bins") or []
            confirmed_blocked = [b for b, c in zip(bins, conf)
                                 if c and int(b) in decision["blocked"]]
            if confirmed_blocked:
                parts.append(f"confirmed={[int(b) for b in confirmed_blocked]}")
    if decision["alert"]:
        parts.append(f"alert={decision['alert']}")
    if decision["closest"] is not None:
        parts.append(f"closest={decision['closest']:.2f}m@{decision['closest_bin']:+d}")
    if decision.get("closest_ttc") is not None:
        parts.append(f"ttc={decision['closest_ttc']:.1f}s@{decision['closest_ttc_bin']:+d}")
    if decision.get("fastest_closing") is not None:
        parts.append(f"v={decision['fastest_closing']:+.1f}m/s@{decision['fastest_bin']:+d}")
    if decision.get("recommended_heading_deg") is not None:
        parts.append(
            f"head={decision['recommended_heading_deg']:+.0f}°({decision['heading_reason']})"
        )
    elif decision.get("protocol", 0) >= 1:
        parts.append("head=?")
    if decision.get("smoothed_heading_deg") is not None:
        parts.append(f"steer={decision['smoothed_heading_deg']:+.0f}°")
    abs_head = absolute_heading(decision, yaw_sign)
    if abs_head is not None:
        parts.append(f"compass={abs_head:+.0f}°")
    fs = decision.get("free_space_m")
    if fs:
        finite = [d for d in fs if d is not None]
        if finite:
            parts.append(f"freespace_min={min(finite):.1f}m")
    if decision["hard_stop"]:
        parts.append("HARD_STOP")
    return "  ".join(parts)


def consume(path: Path, blocked_threshold: float, hard_stop_m: float | None,
            quiet: bool = False, yaw_sign: float = 1.0) -> dict:
    n = 0
    n_blocked = 0
    n_hard_stop = 0
    n_alert = 0
    n_heading = 0
    n_heading_ambiguous = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            decision = decide(rec, blocked_threshold, hard_stop_m)
            n += 1
            if decision["blocked"]:
                n_blocked += 1
            if decision["alert"]:
                n_alert += 1
            if decision["hard_stop"]:
                n_hard_stop += 1
            if decision.get("protocol", 0) >= 1:
                if decision.get("recommended_heading_deg") is not None:
                    n_heading += 1
                else:
                    n_heading_ambiguous += 1
            if not quiet and (decision["blocked"] or decision["hard_stop"]):
                print(format_line(decision, yaw_sign))
    return {
        "n_frames": n,
        "n_blocked": n_blocked,
        "n_alert": n_alert,
        "n_hard_stop": n_hard_stop,
        "n_heading": n_heading,
        "n_heading_ambiguous": n_heading_ambiguous,
        "blocked_rate": n_blocked / n if n else 0.0,
        "hard_stop_rate": n_hard_stop / n if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to a sectors_*.jsonl file.")
    ap.add_argument("--blocked", type=float, default=0.66,
                    help="Score threshold for treating a sector as blocked.")
    ap.add_argument("--hard-stop-m", type=float, default=None,
                    help="If set, flag frames where the closest radar return "
                         "is within this many metres.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-frame lines; only print the summary.")
    ap.add_argument("--yaw-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="Sign relating IMU yaw to starboard-positive steer "
                         "when composing the compass heading: VERIFY on the "
                         "boat against a landmark before nav use.")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    stats = consume(path, args.blocked, args.hard_stop_m, args.quiet,
                    yaw_sign=args.yaw_sign)

    print()
    print(f"frames           : {stats['n_frames']}")
    print(f"blocked frames   : {stats['n_blocked']} ({stats['blocked_rate']:.1%})")
    print(f"alert  frames    : {stats['n_alert']}")
    if args.hard_stop_m is not None:
        print(f"hard-stop frames : {stats['n_hard_stop']} ({stats['hard_stop_rate']:.1%})")
    if stats["n_heading"] or stats["n_heading_ambiguous"]:
        recommended = stats["n_heading"]
        ambiguous = stats["n_heading_ambiguous"]
        total = recommended + ambiguous
        print(f"heading frames   : {recommended} recommended / {ambiguous} ambiguous "
              f"({recommended/total:.1%} recommended of v1 frames)")


if __name__ == "__main__":
    main()
