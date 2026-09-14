"""Enrich a baked bundle with environmental context for the data-completion
map (and, downstream, the fusion-scorer context features):

- gps_init:    static capture position until live GPS lands (source-tagged)
- sun:         per-clip sun-path samples (elevation/azimuth, NOAA solar
               geometry, Europe/Berlin DST-aware) + daypart classification
- luminance:   per-frame mean luma sampled from the baked JPEGs -> histogram
- weather:     Open-Meteo archive (cloud cover, precipitation, wind, temp);
               honest nulls when offline / not yet archived
- entities:    instance + frame counts per class from boxes/instances/labels

Writes an `enrichment` block into each clip's meta.json AND clips.json
summary (the map page reads the catalogue only).

    .venv/bin/python dashboard/tools/enrich_bundle.py [--bundle DIR] [--offline]
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

BUNDLE = Path("/Volumes/ROS2_SSD/asvproject/dashboard_bundle")
TZ = ZoneInfo("Europe/Berlin")

# 47°41'44.3"N 9°11'38.0"E — the capture site, used for sun geometry and the
# weather lookup for every clip in the bundle.
#
# Live GPS reached CAPTURE on 2026-08-24: clips recorded from then on carry a
# per-clip `gps_<ts>.csv` sidecar (`datasets.load_gps_csv`). The baker still
# uses this one constant, because the whole corpus predates the sidecar and one
# static point is right for all of it — the lake is small enough that solar
# elevation varies well under a degree across it. Wiring the sidecar in is a
# per-clip refinement, not a correction; when someone does, honour its `Source`
# column (a `fallback` row is this same guess wearing per-clip coordinates).
#
# NB `configs/gps.yaml` holds the same point as the capture-side fallback, at
# 9.193917 rather than 9.193889 — 2 m apart, i.e. the same place read off the
# map twice. Neither is wrong and neither matters here; don't "fix" one into
# the other believing you have found a bug.
GPS_INIT = {"lat": 47.695639, "lon": 9.193889, "source": "static_init"}

LUM_SAMPLE_EVERY = 3
LUM_BINS = 16  # 0..255 in 16-wide bins
SUN_SAMPLE_S = 60


def solar_position(dt_utc: datetime, lat: float, lon: float) -> tuple[float, float]:
    """NOAA solar geometry -> (elevation_deg, azimuth_deg from N, cw).
    Accuracy well under 0.5 deg; plenty for coverage mapping."""
    ts = dt_utc.timestamp() / 86400.0 + 2440587.5  # julian day
    d = ts - 2451545.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.degrees(math.atan2(math.cos(e) * math.sin(L), math.cos(L))) % 360
    dec = math.asin(math.sin(e) * math.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    lst = (gmst * 15 + lon) % 360
    ha = math.radians((lst - ra + 540) % 360 - 180)
    lat_r = math.radians(lat)
    elev = math.asin(math.sin(lat_r) * math.sin(dec)
                     + math.cos(lat_r) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha),
                    math.tan(dec) * math.cos(lat_r) - math.sin(lat_r) * math.cos(ha))
    return math.degrees(elev), math.degrees(az) % 360


def daypart(elev: float) -> str:
    if elev < -6:
        return "night"
    if elev < 0:
        return "twilight"
    if elev < 10:
        return "golden"
    return "day"


def parse_frame_dt(date_str: str, ts: str) -> datetime:
    hh, mm, rest = ts.split(":")
    return datetime(
        int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]),
        int(hh), int(mm), int(float(rest)), tzinfo=TZ)


def fetch_weather(date_str: str, offline: bool) -> dict | None:
    """Hourly weather for the capture day at GPS_INIT (Open-Meteo, keyless).
    Archive first; recent days fall back to the forecast API's past window."""
    if offline:
        return None
    hourly = ("temperature_2m,cloud_cover,precipitation,"
              "wind_speed_10m,wind_direction_10m")
    for base, extra in (
        ("https://archive-api.open-meteo.com/v1/archive",
         {"start_date": date_str, "end_date": date_str}),
        ("https://api.open-meteo.com/v1/forecast",
         {"past_days": "92", "forecast_days": "1"}),
    ):
        params = {"latitude": GPS_INIT["lat"], "longitude": GPS_INIT["lon"],
                  "hourly": hourly, "timezone": "Europe/Berlin", **extra}
        try:
            with urllib.request.urlopen(
                    f"{base}?{urllib.parse.urlencode(params)}", timeout=15) as r:
                data = json.load(r)
            h = data["hourly"]
            idx = [i for i, t in enumerate(h["time"]) if t.startswith(date_str)]
            if not idx or h["cloud_cover"][idx[len(idx) // 2]] is None:
                continue
            return {
                "hours": [int(h["time"][i][11:13]) for i in idx],
                "cloud_cover": [h["cloud_cover"][i] for i in idx],
                "precipitation": [h["precipitation"][i] for i in idx],
                "wind_speed": [h["wind_speed_10m"][i] for i in idx],
                "wind_dir": [h.get("wind_direction_10m", [None] * len(h["time"]))[i]
                             for i in idx],
                "temperature": [h["temperature_2m"][i] for i in idx],
                "source": "open-meteo",
            }
        except Exception:
            continue
    return None


def enrich_clip(clip_dir: Path, weather_cache: dict, offline: bool) -> dict:
    meta = json.loads((clip_dir / "meta.json").read_text())
    date_str = meta["triplet_ts"][:10]
    frames = meta["frames"]

    # --- sun path, sampled every SUN_SAMPLE_S along the activity
    samples = []
    last = None
    for f in frames:
        dt = parse_frame_dt(date_str, f["ts"])
        if last is not None and (dt - last).total_seconds() < SUN_SAMPLE_S:
            continue
        last = dt
        elev, az = solar_position(dt.astimezone(timezone.utc),
                                  GPS_INIT["lat"], GPS_INIT["lon"])
        samples.append([f["ts"], round(elev, 1), round(az, 1)])
    dayparts: dict[str, int] = {}
    for f in frames:
        dt = parse_frame_dt(date_str, f["ts"])
        # nearest sample is fine at 60 s granularity
        elev = min(samples, key=lambda s: abs(
            (parse_frame_dt(date_str, s[0]) - dt).total_seconds()))[1]
        dayparts[daypart(elev)] = dayparts.get(daypart(elev), 0) + 1

    # --- luminance from the baked frames
    hist = [0] * LUM_BINS
    lum_sum = 0.0
    lum_n = 0
    for f in frames[::LUM_SAMPLE_EVERY]:
        img = cv2.imread(
            str(clip_dir / "frames" / f"ts={f['ts'].replace(':', '-')}.jpg"),
            cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        m = float(img.mean())
        hist[min(int(m / 256 * LUM_BINS), LUM_BINS - 1)] += 1
        lum_sum += m
        lum_n += 1

    # --- entity instances, kept PER DETECTION STREAM. The three artefact
    # files are different detectors over the same frames (and the quad
    # clip's boxes/labels are literally the same GroundingDINO records), so
    # summing them would overcount physical instances up to 3x. Headline =
    # max across streams; the full split ships for provenance display.
    STREAMS = {"boxes.json": "typed-det", "instances.json": "instance-seg",
               "labels.json": "labels"}
    per_stream: dict[str, dict[str, dict[str, int]]] = {}
    for name, stream in STREAMS.items():
        path = clip_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        agg: dict[str, dict[str, int]] = {}
        for ts, items in data.items():
            if name == "labels.json":
                items = items.get("fisheye_bboxes", [])
            for it in items:
                cls = it.get("cls", "unknown")
                e = agg.setdefault(cls, {"instances": 0, "frames": 0})
                e["instances"] += 1
            for cls in {it.get("cls", "unknown") for it in items}:
                agg.setdefault(cls, {"instances": 0, "frames": 0})
                agg[cls]["frames"] += 1
        if agg:
            per_stream[stream] = agg
    entities: dict[str, dict] = {}
    for stream, agg in per_stream.items():
        for cls, e in agg.items():
            ent = entities.setdefault(
                cls, {"instances": 0, "frames": 0, "by_source": {}})
            ent["by_source"][stream] = e
            if e["instances"] > ent["instances"]:
                ent["instances"] = e["instances"]
                ent["frames"] = e["frames"]

    # --- weather (cached per day)
    if date_str not in weather_cache:
        weather_cache[date_str] = fetch_weather(date_str, offline)
    weather = weather_cache[date_str]
    weather_clip = None
    if weather:
        start_h = int(frames[0]["ts"][:2])
        end_h = int(frames[-1]["ts"][:2])
        sel = [i for i, h in enumerate(weather["hours"]) if start_h <= h <= end_h]
        if sel:
            dirs = [weather.get("wind_dir", [None])[i] for i in sel
                    if weather.get("wind_dir") and weather["wind_dir"][i] is not None]
            wind_dir = None
            if dirs:
                # circular mean: wind directions cannot be averaged linearly
                sx = sum(math.cos(math.radians(d)) for d in dirs)
                sy = sum(math.sin(math.radians(d)) for d in dirs)
                wind_dir = round(math.degrees(math.atan2(sy, sx)) % 360)
            weather_clip = {
                "cloud_cover_pct": round(sum(weather["cloud_cover"][i] for i in sel) / len(sel)),
                "precipitation_mm": round(sum(weather["precipitation"][i] for i in sel), 2),
                "wind_speed_kmh": round(sum(weather["wind_speed"][i] for i in sel) / len(sel), 1),
                "wind_dir_deg": wind_dir,
                "temperature_c": round(sum(weather["temperature"][i] for i in sel) / len(sel), 1),
                "source": weather["source"],
            }

    enrichment = {
        "gps_init": GPS_INIT,
        "sun_samples": samples,          # [ts, elevation_deg, azimuth_deg]
        "dayparts": dayparts,            # frames per night/twilight/golden/day
        "luminance": {
            "mean": round(lum_sum / lum_n, 1) if lum_n else None,
            "hist": hist,                # LUM_BINS bins over 0..255, sampled
            "sampled_every": LUM_SAMPLE_EVERY,
        },
        "weather": weather_clip,         # None = honestly unknown
        "entities": entities,
    }
    meta["enrichment"] = enrichment
    (clip_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return enrichment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    ap.add_argument("--offline", action="store_true",
                    help="skip the weather fetch (fields stay null)")
    args = ap.parse_args()

    catalogue_path = args.bundle / "clips.json"
    catalogue = json.loads(catalogue_path.read_text())
    weather_cache: dict = {}
    for clip in catalogue["clips"]:
        key = clip["clip_id"].replace("/", "__")
        enrichment = enrich_clip(args.bundle / key, weather_cache, args.offline)
        clip["enrichment"] = enrichment
        sun = enrichment["sun_samples"]
        print(f"  {clip['clip_id']}: sun elev {sun[0][1]}→{sun[-1][1]}°, "
              f"lum {enrichment['luminance']['mean']}, "
              f"weather {'ok' if enrichment['weather'] else 'null'}, "
              f"entities {sum(e['instances'] for e in enrichment['entities'].values())}")
    catalogue_path.write_text(json.dumps(catalogue, separators=(",", ":")))
    print(f"enriched {len(catalogue['clips'])} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
