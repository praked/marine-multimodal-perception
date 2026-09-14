# Extrinsics: photogrammetric in-plane measurements (2026-07-07)

Source: IMG_9505 (fronto-parallel lid photo) + calliper reference: bolt
outer-outer spans 107.5 mm ("length" pairs) x 60.3 mm ("height" pairs).
Method: homography from the four M3 bolt-cap centres (cap Ø derived
5.36 mm) to the known rectangle; scales agreed to 0.16 % across axes.
Estimated accuracy ±1 mm. Overlay: IMG_9505_measured.jpg (Downloads).

Reference points: radar = centre of the antenna patch region (beige
block); fisheye = lens barrel centre; thermal = the Lepton lens dome
(the small metallic dome, offset right inside the sealed window).

Box axes: X along the 60.3 axis, Y along the 107.5 axis
(Y+ = from thermal end toward fisheye end, as photographed).

| pair              | dX (mm) | dY (mm) | straight-line |
|-------------------|--------:|--------:|--------------:|
| fisheye -> radar  |  +51.6  |   -2.0  | 51.6 |
| thermal -> radar  |  +50.4  |  +60.4  | 78.6 |
| fisheye -> thermal|   +1.2  |  -62.3  | 62.4 |

Consistency: (fisheye->thermal) + (thermal->radar) = (+51.6, -2.0) ✓.

STILL NEEDED before extrinsics.yaml can be filled:
1. Deployed orientation (which photo edge faces up on the boat): sets
   the ± sign mapping to right/up.
2. The two FORWARD (depth) offsets, by hand: radar board face ->
   fisheye lens tip, and -> Lepton window (a few mm each).
3. Confirm the 107.5/60.3 were calliper'd on the METAL caps (not the
   rubber grommets): extraction assumed metal caps.
4. Physical sanity check (pending).

## RESOLVED (same day): extrinsics.yaml is live

Physical sanity check passed (51.6 mm exact; 62.4 measured ~63.5;
cap Ø exact). Orientation confirmed: 60.3 axis vertical, antennas up,
fisheye above thermal. Approximate depths: lens tips 4.3 / 5.0 mm
forward of the radar board face. configs/extrinsics.yaml now carries
measured translations (`measured: true`), thermal camera height split
to 0.208 m, and a translation table for all four 90° remounts.
Verification: full suite green (624); a dead-ahead radar point at 2 m
projects 10.8 px left of the fisheye centre vs 10.6 px predicted.
V.1 is closed; Branch A.1 cross-sensor projection is unblocked.
