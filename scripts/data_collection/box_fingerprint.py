#!/usr/bin/env python3
"""Machine-readable state fingerprint of a capture box, and a diff between two.

Written for the 2026-08-20 duplication: the Pi 4 is imaged, the image is
flashed to a card for the Pi 5, and the claim to verify is "nothing changed".
That claim is unfalsifiable by eye. A naive diff is no better, because a board
swap *must* change some things (model, serial, kernel flavour, MAC, the
board-specific UART overlay) and a second box *must* change others (hostname,
machine-id, host keys). Drowning the real differences in expected ones is how
a genuine regression gets waved through.

So every field is classified:

    board      expected to differ across a Pi 4 -> Pi 5 swap
    identity   expected to differ between two distinct boxes
    substance  must match; a difference here is a regression

`--compare` reports the three separately and exits non-zero only on substance.

Stdlib only, so it runs on a freshly flashed card before anything is installed.

    python3 -m scripts.data_collection.box_fingerprint -o before.json
    python3 -m scripts.data_collection.box_fingerprint --compare before.json after.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Fields a Pi 4 -> Pi 5 swap is allowed to change.
BOARD_KEYS = {
    "hw.model", "hw.revision", "hw.serial", "os.kernel", "os.kernel_flavour",
    "hw.cpu_cores", "hw.memory_kb", "boot.uart_lines", "power.throttled",
    "dev.video_nodes",
}
# Fields two distinct boxes are allowed to differ on.
IDENTITY_KEYS = {
    "id.hostname", "id.machine_id", "id.ssh_host_key_sha256", "net.mac_eth0",
    "net.mac_wlan0", "net.tailscale_ip", "net.ipv4", "id.boot_id",
}

# Config files whose *content* is the thing under test.
TRACKED_FILES = [
    ("cfg.box_env", "/etc/asvproject/box.env"),
    ("cfg.radar_default", "~/test_config.cfg"),
    ("cfg.radar_clutter_on", "~/test_config_clutter_on.cfg"),
    ("cfg.radar_cfar8", "~/test_config_cfar8.cfg"),
    ("cfg.radar_cfar15", "~/test_config_cfar15.cfg"),
    ("cfg.capture_service", "/etc/systemd/system/asvproject-capture.service"),
]
# Repo configs: the calibration itself. A silent edit here is the worst case.
REPO_CONFIGS = ["detection.yaml", "intrinsics.yaml", "extrinsics.yaml",
                "imu.yaml", "clip_overrides.yaml"]

REPO = Path(os.environ.get("ASVPROJECT_REPO",
                           Path.home() / "ASVProject-ObstacleDetection"))


def _run(*cmd: str, timeout: float = 15.0) -> str | None:
    """Best-effort command output; None when the tool is absent or fails.

    A probe that cannot answer records None rather than raising: a fingerprint
    of a half-built card is still worth having.
    """
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _ok(*cmd: str, timeout: float = 15.0) -> bool:
    """Did the command succeed? Distinct from _run, which reports stdout and
    so cannot tell success-with-no-output (`sudo -n true`) from failure."""
    if shutil.which(cmd[0]) is None:
        return False
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout,
                              text=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _read(path: str) -> str | None:
    try:
        return Path(path).expanduser().read_text(errors="replace")
    except OSError:
        return None


def _sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _file_digest(path: str) -> str | None:
    return _sha256(_read(path))


def _proc_cpuinfo(field: str) -> str | None:
    txt = _read("/proc/cpuinfo") or ""
    for line in txt.splitlines():
        if line.lower().startswith(field.lower()):
            return line.split(":", 1)[1].strip()
    return None


def _config_txt() -> tuple[str | None, str | None]:
    """(digest of the meaningful lines, the UART/board-specific lines).

    Comments and blank lines are dropped so a reformat is not a regression.
    The UART lines are split out because they are exactly what must differ
    between a Pi 4 (`dtoverlay=miniuart-bt` + the `core_freq` pins it needs;
    `disable-bt` before 2026-09-03) and a Pi 5 (`dtparam=uart0=on`).
    """
    txt = _read("/boot/firmware/config.txt")
    if txt is None:
        return None, None
    lines = [ln.strip() for ln in txt.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    uart = sorted(ln for ln in lines
                  if re.search(r"uart|disable-bt|miniuart|core_freq", ln, re.I))
    rest = sorted(ln for ln in lines if ln not in uart)
    return _sha256("\n".join(rest)), "; ".join(uart) or None


def _pip_freeze() -> str | None:
    for cmd in (("pip3", "freeze"), ("pip", "freeze")):
        out = _run(*cmd, timeout=60)
        if out:
            return _sha256("\n".join(sorted(out.splitlines())))
    return None


def _apt_manifest() -> str | None:
    out = _run("dpkg-query", "-W", "-f=${Package}=${Version}\n", timeout=60)
    return _sha256("\n".join(sorted(out.splitlines()))) if out else None


def sshd_effective(keyword: str) -> str | None:
    """Effective sshd setting, or None if it genuinely cannot be determined.

    Ask sshd itself. It resolves `Include` and, crucially, sshd's
    first-value-wins precedence, which reading the files naively gets
    backwards. The drop-ins are mode 600, so an unprivileged read sees neither
    the cloud-init default nor anything added below it; `sudo -n` is tried
    because these boxes keep passwordless sudo for exactly this kind of probe.
    """
    for cmd in (("sshd", "-T"), ("/usr/sbin/sshd", "-T"),
                ("sudo", "-n", "sshd", "-T"),
                ("sudo", "-n", "/usr/sbin/sshd", "-T")):
        out = _run(*cmd, timeout=20)
        if not out:
            continue
        for line in out.splitlines():
            parts = line.split(None, 1)
            if parts and parts[0].lower() == keyword.lower():
                return parts[1].strip().lower() if len(parts) > 1 else ""
    return None


def _password_auth() -> str:
    """Whether SSH accepts passwords. 'unknown' when it cannot be read.

    Reporting a confident wrong answer here is worse than reporting none: this
    field exists to track the box's auth posture, and it changed the day it
    was added without the diff noticing (2026-08-21).
    """
    eff = sshd_effective("PasswordAuthentication")
    if eff is not None:
        return eff
    # Fallback: parse what we can read, honouring FIRST-match precedence.
    chunks = [_read("/etc/ssh/sshd_config")]
    d = Path("/etc/ssh/sshd_config.d")
    readable = True
    if d.is_dir():
        for conf in sorted(d.glob("*.conf")):
            body = _read(str(conf))
            readable &= body is not None
            chunks.append(body)
    text = "\n".join(c for c in chunks if c)
    m = re.search(r"(?mi)^\s*PasswordAuthentication\s+(\w+)", text)
    if m:
        return m.group(1).lower()
    # No match AND we could not read every file: we do not actually know.
    return "default(no)" if readable else "unknown (unreadable)"


def _scripts_digest() -> dict:
    """Content digest of the deployed Python under scripts/.

    The box is an rsync copy, not a git clone, so repo.commit is null there and
    a hand-edited script on the Pi would otherwise pass "nothing changed"
    unnoticed. This is the field that actually pins the deployed code.
    """
    root = REPO / "scripts"
    if not root.is_dir():
        return {"repo.scripts_digest": None, "repo.scripts_files": 0}
    entries = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            body = path.read_bytes()
        except OSError:
            continue
        entries.append(f"{path.relative_to(root)}:"
                       f"{hashlib.sha256(body).hexdigest()}")
    return {"repo.scripts_digest": _sha256("\n".join(entries)),
            "repo.scripts_files": len(entries)}


def _repo_state() -> dict:
    if not (REPO / ".git").exists():
        return {"repo.commit": None, "repo.branch": None, "repo.dirty": None}
    def g(*a):
        try:
            r = subprocess.run(("git", "-C", str(REPO)) + a,
                               capture_output=True, text=True, timeout=20)
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "repo.commit": g("rev-parse", "HEAD"),
        "repo.branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        # Uncommitted edits on the Pi are how deployments drift from the repo.
        "repo.dirty": bool(g("status", "--porcelain")),
    }


def _model_digests() -> dict:
    """Hashes of the deployed ONNX/pth models under ~/ewasr.

    Large files, so hash by size+name rather than content: a full content hash
    of 516 MB on a Pi 4 is slow enough that people skip running this at all.
    """
    root = Path.home() / "ewasr"
    if not root.is_dir():
        return {"models.ewasr": None, "models.ewasr_count": 0}
    entries = sorted(
        f"{p.relative_to(root)}:{p.stat().st_size}"
        for p in root.rglob("*") if p.is_file())
    return {"models.ewasr": _sha256("\n".join(entries)),
            "models.ewasr_count": len(entries)}


def _systemd() -> dict:
    out = {}
    for unit in ("asvproject-capture", "asvproject-bt-up", "ssh", "tailscaled"):
        state = _run("systemctl", "is-enabled", unit) or "absent"
        out[f"systemd.{unit}"] = state
    return out


def _thermal_by_id() -> str | None:
    d = Path("/dev/v4l/by-id")
    if not d.is_dir():
        return None
    hits = sorted(p.name for p in d.iterdir() if "PureThermal" in p.name)
    return "; ".join(hits) or None


def collect() -> dict:
    """Probe the box. Every value is a scalar so the diff stays readable."""
    cfg_digest, uart_lines = _config_txt()
    fp: dict = {
        # --- board ---
        "hw.model": (_read("/proc/device-tree/model") or "").strip("\x00 ") or None,
        "hw.revision": _proc_cpuinfo("Revision"),
        "hw.serial": _proc_cpuinfo("Serial"),
        "hw.cpu_cores": os.cpu_count(),
        "hw.memory_kb": next((int(l.split()[1]) for l in
                              (_read("/proc/meminfo") or "").splitlines()
                              if l.startswith("MemTotal")), None),
        "os.kernel": (_run("uname", "-r") or None),
        "os.kernel_flavour": ("pi5" if "2712" in (_run("uname", "-r") or "")
                              else "pi4"),
        "os.release": next((l.split("=", 1)[1].strip('"') for l in
                            (_read("/etc/os-release") or "").splitlines()
                            if l.startswith("PRETTY_NAME=")), None),
        "os.arch": _run("dpkg", "--print-architecture"),
        "power.throttled": _run("vcgencmd", "get_throttled"),

        # --- identity ---
        "id.hostname": _run("hostname"),
        "id.machine_id": (_read("/etc/machine-id") or "").strip() or None,
        "id.ssh_host_key_sha256": _file_digest(
            "/etc/ssh/ssh_host_ed25519_key.pub"),
        "net.tailscale_ip": _run("tailscale", "ip", "-4"),

        # --- substance: boot + devices ---
        "boot.config_digest": cfg_digest,
        "boot.uart_lines": uart_lines,
        "boot.cmdline_digest": _file_digest("/boot/firmware/cmdline.txt"),
        "dev.ttyAMA0": Path("/dev/ttyAMA0").exists(),
        "dev.rtc0": Path("/dev/rtc0").exists(),
        # WHICH chip owns rtc0. On a Pi 5 the built-in RTC can claim it and
        # push the external DS3231 to rtc1, silently changing which clock the
        # kernel restores at boot. The built-in one has no backup cell fitted
        # (2026-07-10), so that swap would break offline time while every
        # other check still passes. Expect "rtc-ds1307" for the DS3231.
        "dev.rtc0_driver": (_read("/sys/class/rtc/rtc0/name") or "").strip()
                           or None,
        "dev.rtc_devices": ",".join(sorted(
            p.name for p in Path("/sys/class/rtc").glob("rtc*"))) or None,
        "dev.i2c1": Path("/dev/i2c-1").exists(),
        "dev.video_nodes": ",".join(sorted(
            p.name for p in Path("/dev").glob("video*"))) or None,
        "dev.thermal_by_id": _thermal_by_id(),
        "dev.radar_cli": Path("/dev/ttyACM0").exists(),
        "dev.radar_data": Path("/dev/ttyACM1").exists(),

        # --- substance: software ---
        "sw.apt_manifest": _apt_manifest(),
        "sw.pip_freeze": _pip_freeze(),
        "sw.python": sys.version.split()[0],
    }
    fp.update(_repo_state())
    fp.update(_scripts_digest())
    fp.update(_model_digests())
    fp.update(_systemd())
    for key, path in TRACKED_FILES:
        fp[key] = _file_digest(path)
    for name in REPO_CONFIGS:
        fp[f"repocfg.{name}"] = _file_digest(str(REPO / "configs" / name))
    # sudo policy: the runbooks and the capture tooling depend on passwordless
    # sudo staying available over a non-interactive ssh. `sudo -n` never
    # prompts, so this is safe to probe unattended.
    fp["auth.sudo_nopasswd"] = _ok("sudo", "-n", "true")
    fp["auth.password_ssh"] = _password_auth()
    return fp


def classify(key: str) -> str:
    if key in BOARD_KEYS:
        return "board"
    if key in IDENTITY_KEYS:
        return "identity"
    return "substance"


def compare(a: dict, b: dict) -> dict[str, list[tuple[str, object, object]]]:
    out: dict[str, list] = {"board": [], "identity": [], "substance": []}
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<missing>"), b.get(key, "<missing>")
        if va != vb:
            out[classify(key)].append((key, va, vb))
    return out


def _print_diff(diff: dict, label_a: str, label_b: str) -> None:
    titles = {
        "substance": ("SUBSTANCE — must match; these are regressions", True),
        "board": ("BOARD — expected to differ across a Pi 4/Pi 5 swap", False),
        "identity": ("IDENTITY — expected to differ between two boxes", False),
    }
    for section in ("substance", "board", "identity"):
        rows = diff[section]
        title, _ = titles[section]
        print(f"\n=== {title} ===")
        if not rows:
            print("  (none)")
            continue
        for key, va, vb in rows:
            print(f"  {key}")
            print(f"      {label_a}: {va}")
            print(f"      {label_b}: {vb}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fingerprint a capture box, or diff two fingerprints.")
    ap.add_argument("-o", "--out", help="write the fingerprint here")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="diff two fingerprint files instead of collecting")
    ap.add_argument("--note", default=None,
                    help="free text recorded with the fingerprint "
                         "(e.g. 'Pi4 full box, before imaging')")
    args = ap.parse_args(argv)

    if args.compare:
        before, after = (json.loads(Path(p).read_text()) for p in args.compare)
        a_note = before.pop("_note", args.compare[0])
        b_note = after.pop("_note", args.compare[1])
        before.pop("_collected_on", None), after.pop("_collected_on", None)
        diff = compare(before, after)
        _print_diff(diff, "before", "after")
        n = len(diff["substance"])
        print(f"\n{'FAIL' if n else 'PASS'}: {n} substantive difference"
              f"{'' if n == 1 else 's'} between:")
        print(f"  before = {a_note}")
        print(f"  after  = {b_note}")
        return 1 if n else 0

    fp = collect()
    if args.note:
        fp["_note"] = args.note
    text = json.dumps(fp, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}  ({len(fp)} fields)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
