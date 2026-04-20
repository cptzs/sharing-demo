"""
Basic Lane Detection (Baseline)
================================
This module demonstrates the simplest approach to lane detection:
  1. Convert to grayscale
  2. Apply Gaussian blur
  3. Run Canny edge detection
  4. Crop a fixed trapezoidal region of interest (ROI)
  5. Use the Hough line transform to get line segments
  6. Filter segments by slope to separate left / right lane lines
  7. Extrapolate each side to a single averaged line

This approach is fast and easy to understand, but it struggles with:
  - Curved roads
  - Shadows and variable lighting
  - Worn or faded lane markings
  - Lane changes / merges

Usage (stand-alone, requires OpenCV):
    python lane_detection_basic.py --image <path-to-image>
    python lane_detection_basic.py --video <path-to-video>
"""

import argparse
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def to_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def detect_edges(blurred: np.ndarray,
                 low_threshold: int = 50,
                 high_threshold: int = 150) -> np.ndarray:
    return cv2.Canny(blurred, low_threshold, high_threshold)


def region_of_interest(edges: np.ndarray) -> np.ndarray:
    """Mask everything outside a fixed trapezoidal ROI."""
    h, w = edges.shape
    mask = np.zeros_like(edges)
    # A simple trapezoid that covers the lower half of the image
    vertices = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.55)),
        (int(w * 0.55), int(h * 0.55)),
        (int(w * 0.95), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def hough_lines(masked_edges: np.ndarray) -> list:
    """Return raw Hough line segments."""
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=300,
    )
    return lines if lines is not None else []


def average_slope_intercept(frame: np.ndarray, lines: list):
    """
    Separate line segments into left / right by slope sign, then
    fit a single averaged line for each side.

    Returns (left_line, right_line) where each is (x1, y1, x2, y2) or None.
    """
    h, w = frame.shape[:2]
    left_fits: list = []
    right_fits: list = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:            # skip vertical segments (undefined slope)
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        # Filter out nearly-horizontal segments (likely noise)
        if abs(slope) < 0.4:
            continue
        if slope < 0:           # negative slope → left lane
            left_fits.append((slope, intercept))
        else:                   # positive slope → right lane
            right_fits.append((slope, intercept))

    def make_line(fits):
        if not fits:
            return None
        avg_slope, avg_intercept = np.mean(fits, axis=0)
        y1 = h
        y2 = int(h * 0.55)
        if avg_slope == 0:
            return None
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return (x1, y1, x2, y2)

    return make_line(left_fits), make_line(right_fits)


def draw_lane_lines(frame: np.ndarray, left_line, right_line) -> np.ndarray:
    line_image = np.zeros_like(frame)
    for line in (left_line, right_line):
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_frame(frame: np.ndarray) -> np.ndarray:
    gray    = to_grayscale(frame)
    blurred = apply_gaussian_blur(gray)
    edges   = detect_edges(blurred)
    masked  = region_of_interest(edges)
    lines   = hough_lines(masked)
    left, right = average_slope_intercept(frame, lines)
    return draw_lane_lines(frame, left, right)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Basic lane detection demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image file")
    group.add_argument("--video", help="Path to a video file (or 0 for webcam)")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image: {args.image}")
        result = process_frame(frame)
        cv2.imshow("Basic Lane Detection", result)
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
            result = process_frame(frame)
            cv2.imshow("Basic Lane Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
