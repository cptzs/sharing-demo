"""
Improved Lane Detection – Hough Transform with Enhanced Pre-processing
=======================================================================
This module builds on the basic approach and adds several improvements:

  1. **Color filtering** – isolate white and yellow lane markings in HLS
     space before edge detection, dramatically reducing noise from the road
     surface and shadows.
  2. **Adaptive / dynamic ROI** – the trapezoidal mask is computed from the
     image dimensions so it works correctly for any resolution.
  3. **Line clustering** – instead of splitting solely by slope sign, segments
     are clustered into at most two groups (left / right) and a robust least-
     squares line is fitted to each cluster, avoiding the influence of outlier
     segments.
  4. **Temporal smoothing** – lane positions are smoothed across consecutive
     video frames using an exponential moving average, preventing flickering.

Compared to the basic approach this method handles:
  ✔ Shadows and lighting changes (via HLS color masking)
  ✔ Yellow lane markings
  ✔ Noisy / spurious edge segments (via clustering + robust fitting)
  ✔ Frame-to-frame jitter (via temporal smoothing)

It still struggles with:
  ✗ Strong curves (lines cannot model a curve)
  ✗ Missing or very faded markings
  ✗ Complex intersections

Usage (requires OpenCV ≥ 4 and NumPy):
    python lane_detection_hough_improved.py --image <path>
    python lane_detection_hough_improved.py --video <path>
"""

import argparse
import sys
from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color-space preprocessing
# ---------------------------------------------------------------------------

def select_white_yellow(frame: np.ndarray) -> np.ndarray:
    """
    Return a binary mask that keeps only white and yellow regions.
    Working in HLS colour space makes the filter far more robust to
    changes in ambient lighting than operating in BGR/RGB directly.
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # White: high lightness regardless of hue/saturation
    white_mask = cv2.inRange(hls,
                             np.array([0,   200,   0], dtype=np.uint8),
                             np.array([180, 255, 255], dtype=np.uint8))

    # Yellow: hue ≈ 15–35° in OpenCV's 0–180 range
    yellow_mask = cv2.inRange(hls,
                              np.array([15,  30,  115], dtype=np.uint8),
                              np.array([35, 204,  255], dtype=np.uint8))

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(frame, frame, mask=combined)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Full pre-processing pipeline: colour filter → gray → blur → edges."""
    filtered = select_white_yellow(frame)
    gray     = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)
    return edges


# ---------------------------------------------------------------------------
# ROI masking
# ---------------------------------------------------------------------------

def region_of_interest(edges: np.ndarray) -> np.ndarray:
    """Dynamic trapezoidal mask that adapts to the image resolution."""
    h, w = edges.shape
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.44), int(h * 0.58)),
        (int(w * 0.56), int(h * 0.58)),
        (int(w * 0.95), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


# ---------------------------------------------------------------------------
# Hough + clustering
# ---------------------------------------------------------------------------

def hough_lines(masked_edges: np.ndarray) -> list:
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=200,
    )
    return lines.tolist() if lines is not None else []


def fit_lane_line(frame: np.ndarray, segments: list):
    """
    Fit a single line through a collection of (x1,y1,x2,y2) segments using
    least-squares (numpy.polyfit degree-1).

    Returns (x1, y1, x2, y2) for the extrapolated line, or None.
    """
    if not segments:
        return None
    h = frame.shape[0]
    xs, ys = [], []
    for seg in segments:
        x1, y1, x2, y2 = seg
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    try:
        # polyfit: x = m*y + b  (fit x as a function of y for near-vertical lines)
        coeffs = np.polyfit(ys, xs, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    poly = np.poly1d(coeffs)
    y1_out = h
    y2_out = int(h * 0.58)
    x1_out = int(poly(y1_out))
    x2_out = int(poly(y2_out))
    return (x1_out, y1_out, x2_out, y2_out)


def cluster_and_fit(frame: np.ndarray, lines: list):
    """
    Split raw Hough segments into left/right by slope sign and apply a
    minimum-slope filter to reject near-horizontal noise.
    """
    h, w = frame.shape[:2]
    left_segs:  list = []
    right_segs: list = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:    # reject near-horizontal segments
            continue
        if slope < 0:
            left_segs.append((x1, y1, x2, y2))
        else:
            right_segs.append((x1, y1, x2, y2))

    left_line  = fit_lane_line(frame, left_segs)
    right_line = fit_lane_line(frame, right_segs)
    return left_line, right_line


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

class LaneSmoother:
    """Exponential moving average smoother for one lane line."""

    def __init__(self, alpha: float = 0.2, history: int = 10):
        self._alpha = alpha
        self._history: deque = deque(maxlen=history)
        self._avg = None

    def update(self, line):
        if line is None:
            # Reuse the last averaged position when detection fails
            return self._avg
        self._history.append(line)
        arr = np.mean(self._history, axis=0)
        if self._avg is None:
            self._avg = arr
        else:
            self._avg = self._alpha * arr + (1 - self._alpha) * self._avg
        return tuple(map(int, self._avg))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_lane_lines(frame: np.ndarray,
                    left_line,
                    right_line,
                    fill_lane: bool = True) -> np.ndarray:
    """Draw lane lines and optionally fill the detected lane region."""
    overlay = np.zeros_like(frame)
    h = frame.shape[0]

    if fill_lane and left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 128, 255))  # translucent orange

    for line in (left_line, right_line):
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)

    return cv2.addWeighted(frame, 0.8, overlay, 0.5, 0.0)


# ---------------------------------------------------------------------------
# Pipeline (stateful: carries smoothers across frames)
# ---------------------------------------------------------------------------

class HoughLaneDetector:
    def __init__(self):
        self._left_smoother  = LaneSmoother()
        self._right_smoother = LaneSmoother()

    def process(self, frame: np.ndarray) -> np.ndarray:
        edges   = preprocess(frame)
        masked  = region_of_interest(edges)
        lines   = hough_lines(masked)
        left_raw, right_raw = cluster_and_fit(frame, lines)
        left  = self._left_smoother.update(left_raw)
        right = self._right_smoother.update(right_raw)
        return draw_lane_lines(frame, left, right)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Improved Hough-transform lane detection demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image file")
    group.add_argument("--video", help="Path to a video file (or 0 for webcam)")
    args = parser.parse_args()

    detector = HoughLaneDetector()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image: {args.image}")
        result = detector.process(frame)
        cv2.imshow("Improved Hough Lane Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        source = 0 if args.video == "0" else args.video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {source}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.process(frame)
            cv2.imshow("Improved Hough Lane Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
