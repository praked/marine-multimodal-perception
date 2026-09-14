# Continuous capture: boot-on-power deployment

`continuous_capture.py` + `asvproject-capture.service` together make
the Pi start recording the moment it has power. This is the day-1
"mount on boat, walk away" mode.

## What you get

- All three sensors opened once at boot.
- Rotating 5-minute triplet chunks written to
  `/home/vesselauser/captures/`, named with the existing convention:
  - `fisheye_<YYYY-MM-DD_HH-MM-SS>.mp4`
  - `thermal_<YYYY-MM-DD_HH-MM-SS>.mp4`
  - `mmwave_<YYYY-MM-DD_HH-MM-SS>.csv`
- Disk-free guard at 2 GB (loop sleeps and waits, no overwriting).
- Errors are caught + logged to the systemd journal. The process
  continues unless systemd hits a hard restart loop.

## Install on the Pi (one-time)

Tar-push the repo to the Pi as in `docs/history/2026-06-17_pi_baseline.md`,
then:

```bash
sudo cp ~/ASVProject-ObstacleDetection/scripts/data_collection/asvproject-capture.service \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable asvproject-capture     # boot-on-power
sudo systemctl start  asvproject-capture     # start now (test)
sudo systemctl status asvproject-capture     # confirm Active: running
```

## Daily operations

```bash
sudo systemctl status   asvproject-capture   # is it running?
sudo systemctl stop     asvproject-capture   # pause
sudo systemctl start    asvproject-capture   # resume
sudo systemctl restart  asvproject-capture   # after config change
sudo systemctl disable  asvproject-capture   # don't start on next boot
sudo journalctl -u asvproject-capture -f     # follow live log
ls -la /home/vesselauser/captures/            # what's been written
```

## Tune knobs without editing the unit

Drop-in override (preferred: survives package upgrades):

```bash
sudo systemctl edit asvproject-capture
```

…opens an editor; paste, e.g.:

```ini
[Service]
Environment=ASVPROJECT_CHUNK_SECONDS=600
Environment=ASVPROJECT_FPS=5.0
```

…save, then `sudo systemctl restart asvproject-capture`.

Available knobs:

| Env var | Default | Meaning |
|---|---|---|
| `ASVPROJECT_CAPTURE_DIR` | `/home/vesselauser/captures` | Where chunks land. |
| `ASVPROJECT_CHUNK_SECONDS` | `300` | Length of each triplet chunk. |
| `ASVPROJECT_FPS` | `3.0` | Target frame rate for cameras. |
| `ASVPROJECT_MIN_FREE_GB` | `2.0` | Sleep + wait when disk is below this. |

## Ingest the chunks after a mission

Same convention as `data/<scene>/`, so:

```bash
# Pull from Pi to laptop
rsync -avz vesselauser@100.64.0.1:/home/vesselauser/captures/ \
    ./data/captures/2026-06-17_lab/

# Or use the ingest tool (validates triplet integrity + sidecars):
python3 -m scripts.data.ingest \
    --source vesselauser@100.64.0.1:/home/vesselauser/captures/ \
    --mission 2026-06-17_lab
```

Then label / replay as usual:

```bash
python3 -m scripts.eval.label_tool \
    --triplet data/captures/2026-06-17_lab/<timestamp>
```

## Known limitations

- A crash mid-chunk loses that chunk's mp4 trailer; the file may be
  unreadable. systemd restarts and the next chunk is fine.
- Own-boat IMU and GPS now have sidecars (`imu_<ts>.csv` since
  2026-07-14, `gps_<ts>.csv` since 2026-08-24). GPS is a ~30-min position
  poll of boat1's autopilot log for training metadata, NOT navigation:
  configured in `configs/gps.yaml`, verified with
  `python3 -m scripts.sensor_processing.gps_boat1 --check`, and it falls
  back to a fixed lake position (flagged as `Source=fallback`) when boat1
  is unreachable.
- `sensorStop` is sent before each config push so the radar starts
  clean, but if the autopilot ever shares the radar (it doesn't
  today), this would clobber it. Coordinate before that day arrives.
- The service runs as `vesselauser`. `/dev/ttyACM*` access depends on
  that user being in the `dialout` (or `plugdev`) group; both
  Raspbian defaults include it.
