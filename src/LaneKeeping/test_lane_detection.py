"""
Unit tests for the three lane-detection modules.

Run with pytest:
    pip install pytest numpy opencv-python
    pytest test_lane_detection.py -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_frame(height=720, width=1280, color=(80, 80, 80)):
    """Create a solid-colour BGR frame."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return frame


def _draw_lane_lines_on_frame(frame, left_x1=None, right_x1=None):
    """
    Draw simple straight white lane lines on a BGR frame so that
    the detectors can find something.

    left_x1  – x at the bottom-left of the left lane marking
    right_x1 – x at the bottom-right of the right lane marking
    """
    h, w = frame.shape[:2]
    if left_x1 is None:
        left_x1 = int(w * 0.25)
    if right_x1 is None:
        right_x1 = int(w * 0.75)

    # Left lane line (runs from bottom-left up to ~60% height)
    cv2.line(frame,
             (left_x1, h),
             (int(w * 0.42), int(h * 0.6)),
             (255, 255, 255), 20)
    # Right lane line
    cv2.line(frame,
             (right_x1, h),
             (int(w * 0.58), int(h * 0.6)),
             (255, 255, 255), 20)
    return frame


# ---------------------------------------------------------------------------
# Import guard – OpenCV may not be available in every CI environment
# ---------------------------------------------------------------------------
cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from lane_detection_basic import (
    to_grayscale,
    apply_gaussian_blur,
    detect_edges,
    region_of_interest,
    hough_lines,
    average_slope_intercept,
    draw_lane_lines,
    process_frame,
)
from lane_detection_hough_improved import (
    select_white_yellow,
    preprocess,
    region_of_interest as roi_improved,
    cluster_and_fit,
    LaneSmoother,
    HoughLaneDetector,
)
from lane_detection_sliding_window import (
    build_warp_matrices,
    warp,
    threshold_frame,
    histogram_peaks,
    sliding_window_search,
    SlidingWindowLaneDetector,
    measure_curvature_real,
    vehicle_offset,
)


# ===========================================================================
# Basic module tests
# ===========================================================================

class TestBasicPreprocessing:
    def test_to_grayscale_shape(self):
        frame = _make_frame()
        gray  = to_grayscale(frame)
        assert gray.ndim == 2
        assert gray.shape == (720, 1280)

    def test_gaussian_blur_shape_preserved(self):
        gray    = np.zeros((480, 640), dtype=np.uint8)
        blurred = apply_gaussian_blur(gray)
        assert blurred.shape == gray.shape

    def test_detect_edges_dtype(self):
        gray   = np.zeros((100, 200), dtype=np.uint8)
        edges  = detect_edges(gray)
        assert edges.dtype == np.uint8
        assert edges.shape == (100, 200)

    def test_region_of_interest_clears_top(self):
        edges  = np.ones((720, 1280), dtype=np.uint8) * 255
        masked = region_of_interest(edges)
        # Top 50% should be zeroed out by the trapezoidal mask
        top_section = masked[:int(720 * 0.50), :]
        assert top_section.sum() == 0

    def test_region_of_interest_preserves_bottom(self):
        edges  = np.ones((720, 1280), dtype=np.uint8) * 255
        masked = region_of_interest(edges)
        # Bottom area should have non-zero pixels
        assert masked[600:, 100:1180].sum() > 0


class TestBasicLineDetection:
    def test_average_slope_intercept_no_lines(self):
        frame = _make_frame()
        left, right = average_slope_intercept(frame, [])
        assert left  is None
        assert right is None

    def test_average_slope_intercept_with_lines(self):
        frame = _make_frame()
        # Fake a left line (negative slope) and a right line (positive slope)
        lines = [
            np.array([[200, 700, 400, 450]]),
            np.array([[800, 700, 700, 450]]),
        ]
        left, right = average_slope_intercept(frame, lines)
        # At least one side should be detected for these clear segments
        assert left is not None or right is not None

    def test_draw_lane_lines_returns_same_shape(self):
        frame  = _make_frame()
        result = draw_lane_lines(frame, (300, 720, 500, 400), (900, 720, 700, 400))
        assert result.shape == frame.shape

    def test_draw_lane_lines_with_none(self):
        frame  = _make_frame()
        result = draw_lane_lines(frame, None, None)
        assert result.shape == frame.shape

    def test_process_frame_output_shape(self):
        frame  = _make_frame()
        result = process_frame(frame)
        assert result.shape == frame.shape


# ===========================================================================
# Improved Hough module tests
# ===========================================================================

class TestImprovedHough:
    def test_select_white_yellow_returns_bgr(self):
        frame  = _make_frame()
        result = select_white_yellow(frame)
        assert result.shape == frame.shape

    def test_select_white_yellow_passes_white(self):
        white_frame = np.full((100, 100, 3), 240, dtype=np.uint8)
        result = select_white_yellow(white_frame)
        # At least some pixels should pass the white filter
        assert result.sum() > 0

    def test_preprocess_returns_binary(self):
        frame  = _make_frame()
        edges  = preprocess(frame)
        assert edges.ndim == 2
        assert edges.dtype == np.uint8

    def test_roi_improved_shape(self):
        edges  = np.ones((720, 1280), dtype=np.uint8) * 255
        masked = roi_improved(edges)
        assert masked.shape == edges.shape

    def test_cluster_and_fit_empty(self):
        frame = _make_frame()
        left, right = cluster_and_fit(frame, [])
        assert left  is None
        assert right is None

    def test_lane_smoother_init(self):
        smoother = LaneSmoother()
        assert smoother.update(None) is None

    def test_lane_smoother_converges(self):
        smoother = LaneSmoother(alpha=0.5)
        line = (300, 720, 500, 400)
        for _ in range(20):
            result = smoother.update(line)
        assert result is not None
        assert len(result) == 4

    def test_hough_lane_detector_output_shape(self):
        frame    = _make_frame()
        detector = HoughLaneDetector()
        result   = detector.process(frame)
        assert result.shape == frame.shape

    def test_hough_lane_detector_with_lane_markings(self):
        frame = _make_frame()
        _draw_lane_lines_on_frame(frame)
        detector = HoughLaneDetector()
        result   = detector.process(frame)
        assert result.shape == frame.shape


# ===========================================================================
# Sliding window module tests
# ===========================================================================

class TestSlidingWindow:
    def test_build_warp_matrices(self):
        M, M_inv = build_warp_matrices()
        assert M.shape     == (3, 3)
        assert M_inv.shape == (3, 3)
        # M @ M_inv gives a homogeneous-scaled identity (not unit identity).
        # Normalise by the (2,2) entry and check it is proportional to identity.
        product = M @ M_inv
        scale = product[2, 2]
        assert abs(scale) > 1e-9, "degenerate warp matrix"
        assert np.allclose(product / scale, np.eye(3), atol=1e-4)

    def test_warp_shape_preserved(self):
        frame   = _make_frame()
        M, _    = build_warp_matrices()
        warped  = warp(frame, M)
        assert warped.shape == frame.shape

    def test_threshold_frame_binary(self):
        frame  = _make_frame()
        binary = threshold_frame(frame)
        assert binary.ndim  == 2
        assert binary.dtype == np.uint8
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})

    def test_histogram_peaks_range(self):
        binary = np.zeros((720, 1280), dtype=np.uint8)
        # Add two bright columns to simulate lane pixel peaks
        binary[360:, 320] = 255
        binary[360:, 960] = 255
        lx, rx = histogram_peaks(binary)
        assert 0 <= lx < 640
        assert 640 <= rx < 1280

    def test_sliding_window_search_returns_fits_or_none(self):
        binary = np.zeros((720, 1280), dtype=np.uint8)
        # Draw two vertical stripes of pixels
        binary[100:720, 310:330] = 255
        binary[100:720, 950:970] = 255
        left_fit, right_fit, out_img = sliding_window_search(binary)
        # With strong pixel columns, fits should be found
        assert left_fit  is not None
        assert right_fit is not None
        assert out_img.shape == (720, 1280, 3)

    def test_sliding_window_empty_binary(self):
        binary = np.zeros((720, 1280), dtype=np.uint8)
        left_fit, right_fit, out_img = sliding_window_search(binary)
        assert left_fit  is None
        assert right_fit is None

    def test_measure_curvature_real_positive(self):
        # A straight line polynomial: A=0, B=0, C=640  → infinite radius
        fit = np.array([0.0, 0.0, 640.0])
        radius = measure_curvature_real(fit, 720)
        assert radius > 0

    def test_vehicle_offset_centre(self):
        # Symmetric lanes → vehicle should be near the centre (offset ≈ 0)
        left_fit  = np.array([0.0, -0.5, 320.0])
        right_fit = np.array([0.0,  0.5, 960.0])
        shape = (720, 1280, 3)
        offset = vehicle_offset(left_fit, right_fit, shape)
        assert abs(offset) < 0.5   # within 0.5 m for symmetric configuration

    def test_sliding_window_detector_output_shape(self):
        frame    = _make_frame()
        detector = SlidingWindowLaneDetector()
        result   = detector.process(frame)
        assert result.shape == frame.shape

    def test_sliding_window_detector_with_lane_markings(self):
        """Detector should produce a result without crashing on synthetic lane markings."""
        frame = _make_frame()
        _draw_lane_lines_on_frame(frame)
        detector = SlidingWindowLaneDetector()
        result   = detector.process(frame)
        assert result.shape == frame.shape
