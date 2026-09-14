import cv2
import numpy as np

from scripts.utils.cv_common import (
    build_blob_detector,
    detect_horizon,
    extract_kp,
    undistort_fisheye,
    undistort_thermal,
)


def test_undistort_fisheye_preserves_shape(intrinsics_real, small_fisheye_frame):
    # Upscale to the calibrated size, then undistort.
    K = intrinsics_real["fisheye"]["K"]
    D = intrinsics_real["fisheye"]["D"]
    img = cv2.resize(small_fisheye_frame, (864, 648))
    out = undistort_fisheye(img, K, D)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_undistort_thermal_preserves_shape(intrinsics_real, small_thermal_frame):
    K = intrinsics_real["thermal"]["K"]
    D = intrinsics_real["thermal"]["D"]
    img = cv2.resize(small_thermal_frame, (160, 120))
    out = undistort_thermal(img, K, D)
    assert out.shape == img.shape


def test_detect_horizon_returns_full_mask_on_blank():
    """A perfectly flat image yields no edges; should fall back to whole mask."""
    img = np.full((50, 80, 3), 128, dtype=np.uint8)
    mask, slope, intercept, conf = detect_horizon(img, ransac_iters=5)
    assert mask.shape == (50, 80)
    assert conf == 0.0
    assert mask.sum() > 0  # fallback covers everything


def test_detect_horizon_finds_a_line():
    rng = np.random.default_rng(42)
    # Make an image with a strong horizontal edge at row 30.
    img = np.zeros((60, 100, 3), dtype=np.uint8)
    img[:30] = 220
    img[30:] = 30
    img = img + rng.integers(-3, 4, size=img.shape, endpoint=False).astype(np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask, slope, intercept, conf = detect_horizon(img, ransac_iters=100, confidence_thresh=0.0)
    assert mask.shape == (60, 100)
    # The strong edge should give nonzero confidence.
    assert conf > 0.0
    # The refit line should sit near the true edge (row 30).
    assert abs(intercept - 30) < 5


def test_detect_horizon_is_deterministic():
    """RANSAC is seeded: identical input -> identical line every call.

    Monocular range is hypersensitive to the horizon row at the low mount
    height, so a jittering horizon made ranges swing wildly between renders.
    """
    rng = np.random.default_rng(7)
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[:40] = 200
    img[40:] = 20
    img = np.clip(img + rng.integers(-4, 5, size=img.shape).astype(np.int16),
                  0, 255).astype(np.uint8)
    results = [detect_horizon(img, ransac_iters=80)[1:] for _ in range(5)]
    for r in results[1:]:
        assert r == results[0]


def test_detect_horizon_fallback_when_confidence_low():
    """Two edge points -> RANSAC can fit but x_span is tiny -> confidence 0."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5, 5] = 255
    img[6, 6] = 255
    mask, slope, intercept, conf = detect_horizon(img, ransac_iters=10,
                                                  confidence_thresh=0.99,
                                                  fallback_row=10)
    assert conf < 0.99
    # Fallback row 10 means rows 10+ are 255.
    assert mask[15, 10] == 255


def test_detect_horizon_grayscale_input():
    """A 2-D (grayscale) frame skips the BGR->gray conversion (line 96)."""
    gray = np.full((50, 80), 128, dtype=np.uint8)
    mask, slope, intercept, conf = detect_horizon(gray, ransac_iters=5)
    assert mask.shape == (50, 80)
    assert conf == 0.0
    assert mask.sum() > 0


def test_detect_horizon_all_vertical_edges_falls_back(monkeypatch):
    """When every Canny edge point shares one column, each RANSAC pair has
    x1 == x2 and is skipped, so best_model stays None (line 123)."""
    import scripts.utils.cv_common as cvc

    # Force an edge image with all bright pixels in a single column so every
    # sampled pair is vertical (x1 == x2) and gets `continue`d, leaving
    # best_model None regardless of how many iterations run.
    def fake_canny(img, low, high):
        edges = np.zeros(img.shape[:2], dtype=np.uint8)
        edges[:, 20] = 255  # one column only
        return edges

    monkeypatch.setattr(cvc.cv2, "Canny", fake_canny)
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    mask, slope, intercept, conf = detect_horizon(
        img, ransac_iters=50, fallback_row=8,
    )
    assert conf == 0.0
    assert slope == 0.0
    assert intercept == 8.0
    # Fallback covers rows >= 8.
    assert mask[8:, :].all()
    assert mask[:8, :].sum() == 0


def test_extract_kp_outputs_lists():
    class FakeKP:
        def __init__(self, x, y, size):
            self.pt = (x, y)
            self.size = size

    kps = [FakeKP(120, 50, 6.0), FakeKP(80, 70, 4.0)]
    coords, sizes, angles = extract_kp(kps, cx=100, pix_deg_ratio=10.0)
    assert coords == [(120, 50), (80, 70)]
    assert sizes == [6.0, 4.0]
    assert angles == [2.0, -2.0]


def test_extract_kp_empty():
    coords, sizes, angles = extract_kp([], cx=50, pix_deg_ratio=1.0)
    assert coords == [] and sizes == [] and angles == []


def test_build_blob_detector_constructs():
    params = {
        "minThreshold": 100,
        "maxThreshold": 200,
        "blobColor": 255,
        "minArea": 10,
    }
    det = build_blob_detector(params)
    assert det is not None
    # Detect a bright square -> at least one keypoint.
    img = np.zeros((50, 50), dtype=np.uint8)
    img[20:30, 20:30] = 255
    kps = det.detect(img)
    assert len(kps) >= 1
