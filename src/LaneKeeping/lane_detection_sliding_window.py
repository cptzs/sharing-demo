"""
Advanced Lane Detection – Sliding Window + Polynomial Curve Fitting
====================================================================
This module implements the approach used in the Udacity Self-Driving Car
Nano-degree project (and widely adopted by the research community):

  1. **Camera calibration / perspective warp** – warp the front-facing camera
     view into a bird's-eye (top-down) perspective.  Lane lines that were
     converging in the original view become roughly parallel, making them
     much easier to fit with a polynomial.
  2. **Binary thresholding** – combine colour-space thresholds (HLS S-channel,
     LAB B-channel for yellow) with a Sobel-x gradient threshold to create a
     robust binary image.
  3. **Histogram peak detection** – sum pixel values vertically on the lower
     half of the warped binary image; the two highest peaks give the starting
     x-positions of the left and right lanes.
  4. **Sliding window search** – starting from the histogram peaks, march a
     fixed-size window upward, re-centering on the mean of detected pixels in
     each window.  All pixels captured inside the windows are the lane pixels.
  5. **Polynomial fitting** – fit a 2nd-degree polynomial (y = Ax² + Bx + C)
     through the lane pixels.  A polynomial models curves correctly while a
     straight line cannot.
  6. **Targeted search** – once a good fit is known, subsequent frames search
     only in a narrow margin around the previous polynomial rather than
     running the full sliding-window search again, dramatically reducing
     computation.
  7. **Curvature and offset** – compute the radius of curvature in metres and
     the lateral offset of the vehicle centre from the lane centre.
  8. **Temporal smoothing** – average polynomial coefficients over recent
     frames.

Advantages over Hough-line methods:
  ✔ Correctly handles curved roads (polynomial model)
  ✔ Robust to partial occlusion / missing markings (pixel-level evidence)
  ✔ Computes physically meaningful curvature and lateral offset
  ✔ Targeted search makes it efficient for video

Limitations:
  ✗ Requires a reasonable perspective-warp calibration
  ✗ Fails on very sharp bends outside the bird's-eye-view field of view
  ✗ Still image-processing-based: struggles in heavy rain / night without
    dedicated preprocessing

Usage (requires OpenCV ≥ 4 and NumPy):
    python lane_detection_sliding_window.py --image <path>
    python lane_detection_sliding_window.py --video <path>
"""

import argparse
import sys
from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Perspective warp helpers
# ---------------------------------------------------------------------------

# Default source / destination quad for perspective warp.
# These values assume a 1280×720 image from a typical dashcam.
# Adjust SRC_POINTS to match the actual lane geometry in your footage.
_DEFAULT_SRC = np.float32([
    [595, 452],
    [685, 452],
    [1127, 720],
    [203, 720],
])
_DEFAULT_DST = np.float32([
    [320, 0],
    [960, 0],
    [960, 720],
    [320, 720],
])


def build_warp_matrices(src: np.ndarray = _DEFAULT_SRC,
                        dst: np.ndarray = _DEFAULT_DST):
    """Return (M, M_inv): forward and inverse perspective-transform matrices."""
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def warp(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Binary thresholding
# ---------------------------------------------------------------------------

def threshold_frame(frame: np.ndarray) -> np.ndarray:
    """
    Combine multiple thresholds into a single binary image.

    Channels used:
      • Sobel-x on L (HLS) – detects vertical edges (lane line sides)
      • S-channel (HLS)    – robust to shadows for white & yellow markings
      • B-channel (LAB)    – especially sensitive to yellow markings
    """
    # --- HLS ---
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel-x on L
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
    sobel_binary = (scaled >= 20) & (scaled <= 100)

    # S channel threshold
    s_binary = (s_channel >= 170) & (s_channel <= 255)

    # --- LAB B-channel (yellow) ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    # Normalise only if yellow is actually present
    if b_channel.max() > 175:
        b_channel = np.uint8(255 * b_channel / b_channel.max())
    b_binary = (b_channel >= 190) & (b_channel <= 255)

    combined = np.zeros_like(s_channel, dtype=np.uint8)
    combined[(s_binary | b_binary) | sobel_binary] = 255
    return combined


# ---------------------------------------------------------------------------
# Sliding window search
# ---------------------------------------------------------------------------

def histogram_peaks(binary_warped: np.ndarray):
    """Return (leftx_base, rightx_base) from histogram of lower half."""
    h, w = binary_warped.shape
    histogram = np.sum(binary_warped[h // 2:, :], axis=0)
    midpoint  = w // 2
    leftx_base  = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)
    return leftx_base, rightx_base


def sliding_window_search(binary_warped: np.ndarray,
                          n_windows: int = 9,
                          margin: int = 100,
                          min_pixels: int = 50):
    """
    Sliding-window lane pixel detection.

    Returns:
        left_fit, right_fit  – 2nd-degree polynomial coefficients [A, B, C]
        out_img              – visualisation image (bird's eye with windows)
    """
    h, w = binary_warped.shape
    out_img = np.dstack([binary_warped] * 3)

    leftx_base, rightx_base = histogram_peaks(binary_warped)

    window_height = h // n_windows
    nonzero   = binary_warped.nonzero()
    nonzeroy  = np.array(nonzero[0])
    nonzerox  = np.array(nonzero[1])

    leftx_current  = leftx_base
    rightx_current = rightx_base

    left_lane_inds:  list = []
    right_lane_inds: list = []

    for win in range(n_windows):
        win_y_low  = h - (win + 1) * window_height
        win_y_high = h - win * window_height

        win_xleft_low   = leftx_current  - margin
        win_xleft_high  = leftx_current  + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw window rectangles on visualisation
        cv2.rectangle(out_img,
                      (win_xleft_low,  win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img,
                      (win_xright_low,  win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

        good_left  = ((nonzeroy >= win_y_low)  & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low)  & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low)  & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left)  >= min_pixels:
            leftx_current  = int(np.mean(nonzerox[good_left]))
        if len(good_right) >= min_pixels:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Colour the detected pixels
    out_img[lefty,  leftx]  = [255, 0,   0]
    out_img[righty, rightx] = [0,   0, 255]

    # Fit 2nd-order polynomial: x = Ay² + By + C
    left_fit  = np.polyfit(lefty,  leftx,  2) if len(leftx)  > 10 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 10 else None

    return left_fit, right_fit, out_img


def targeted_search(binary_warped: np.ndarray,
                    left_fit: np.ndarray,
                    right_fit: np.ndarray,
                    margin: int = 80):
    """
    Fast search in a margin around the previous polynomial fit.
    Returns updated left_fit, right_fit (falls back to sliding window on failure).
    """
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    def _poly_x(fit, y):
        return fit[0] * y**2 + fit[1] * y + fit[2]

    left_lane_inds  = (nonzerox > (_poly_x(left_fit,  nonzeroy) - margin)) & \
                      (nonzerox < (_poly_x(left_fit,  nonzeroy) + margin))
    right_lane_inds = (nonzerox > (_poly_x(right_fit, nonzeroy) - margin)) & \
                      (nonzerox < (_poly_x(right_fit, nonzeroy) + margin))

    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) < 10 or len(rightx) < 10:
        return sliding_window_search(binary_warped)[:2] + (None,)

    left_fit_new  = np.polyfit(lefty,  leftx,  2)
    right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, None


# ---------------------------------------------------------------------------
# Curvature and offset
# ---------------------------------------------------------------------------

# Conversion factors (approximate for typical highway footage, 1280×720)
YM_PER_PIX = 30.0 / 720   # metres per pixel in y
# 700 px: the horizontal span between the two lane lines in the warped
# (bird's-eye) image for a standard 3.7 m lane width at 1280×720.
LANE_WIDTH_PIXELS = 700
XM_PER_PIX = 3.7 / LANE_WIDTH_PIXELS  # metres per pixel in x


def measure_curvature_real(fit_pixel: np.ndarray,
                           image_height: int) -> float:
    """Radius of curvature in metres at the bottom of the image."""
    # Re-fit in world space
    ploty = np.linspace(0, image_height - 1, image_height)
    fitx  = fit_pixel[0] * ploty**2 + fit_pixel[1] * ploty + fit_pixel[2]
    fit_cr = np.polyfit(ploty * YM_PER_PIX, fitx * XM_PER_PIX, 2)
    y_eval = image_height * YM_PER_PIX
    curvature = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1])**2) ** 1.5) / \
                (abs(2 * fit_cr[0]) + 1e-6)
    return curvature


def vehicle_offset(left_fit: np.ndarray,
                   right_fit: np.ndarray,
                   image_shape) -> float:
    """Lateral offset of vehicle centre from lane centre, in metres."""
    h, w = image_shape[:2]
    y_bottom = h - 1
    left_x  = left_fit[0]  * y_bottom**2 + left_fit[1]  * y_bottom + left_fit[2]
    right_x = right_fit[0] * y_bottom**2 + right_fit[1] * y_bottom + right_fit[2]
    lane_centre   = (left_x + right_x) / 2
    camera_centre = w / 2
    return (camera_centre - lane_centre) * XM_PER_PIX


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_lane_area(original: np.ndarray,
                   binary_warped: np.ndarray,
                   left_fit: np.ndarray,
                   right_fit: np.ndarray,
                   M_inv: np.ndarray) -> np.ndarray:
    """Project the fitted lane polygon back onto the original (un-warped) frame."""
    h, w = binary_warped.shape
    ploty  = np.linspace(0, h - 1, h)
    left_x  = left_fit[0]  * ploty**2 + left_fit[1]  * ploty + left_fit[2]
    right_x = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    pts_left  = np.array([np.transpose(np.vstack([left_x,  ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
    pts       = np.hstack((pts_left, pts_right))

    color_warp = np.zeros_like(binary_warped)
    color_warp_3ch = np.dstack([color_warp] * 3)
    cv2.fillPoly(color_warp_3ch, np.int32([pts]), (0, 255, 0))

    # Also draw the actual lane lines
    for x_vals in (left_x, right_x):
        pts_line = np.int32(np.column_stack([x_vals, ploty]))
        cv2.polylines(color_warp_3ch, [pts_line], False, (255, 255, 0), thickness=15)

    newwarp = warp(color_warp_3ch, M_inv)
    return cv2.addWeighted(original, 1, newwarp, 0.3, 0)


def annotate(frame: np.ndarray, curvature_m: float, offset_m: float) -> np.ndarray:
    direction = "right" if offset_m > 0 else "left"
    cv2.putText(frame,
                f"Radius of Curvature: {curvature_m:.0f} m",
                (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame,
                f"Vehicle is {abs(offset_m):.2f} m {direction} of centre",
                (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return frame


# ---------------------------------------------------------------------------
# Pipeline (stateful)
# ---------------------------------------------------------------------------

class SlidingWindowLaneDetector:
    """Full bird's-eye-view + polynomial-fit lane detector."""

    def __init__(self,
                 src: np.ndarray = _DEFAULT_SRC,
                 dst: np.ndarray = _DEFAULT_DST,
                 smooth_frames: int = 5):
        self._M, self._M_inv = build_warp_matrices(src, dst)
        self._left_fit:  np.ndarray | None = None
        self._right_fit: np.ndarray | None = None
        self._left_history:  deque = deque(maxlen=smooth_frames)
        self._right_history: deque = deque(maxlen=smooth_frames)

    def _smooth_fit(self, history: deque, new_fit) -> np.ndarray | None:
        if new_fit is not None:
            history.append(new_fit)
        if not history:
            return None
        return np.mean(history, axis=0)

    def process(self, frame: np.ndarray) -> np.ndarray:
        warped  = warp(frame, self._M)
        binary  = threshold_frame(warped)

        if self._left_fit is None or self._right_fit is None:
            left_fit, right_fit, _ = sliding_window_search(binary)
        else:
            left_fit, right_fit, _ = targeted_search(
                binary, self._left_fit, self._right_fit)

        left_fit  = self._smooth_fit(self._left_history,  left_fit)
        right_fit = self._smooth_fit(self._right_history, right_fit)

        self._left_fit  = left_fit
        self._right_fit = right_fit

        if left_fit is None or right_fit is None:
            # Couldn't detect lanes — return original frame with a warning
            cv2.putText(frame, "Lane not detected",
                        (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return frame

        result = draw_lane_area(frame, binary, left_fit, right_fit, self._M_inv)
        curv   = measure_curvature_real(
            (left_fit + right_fit) / 2, binary.shape[0])
        off    = vehicle_offset(left_fit, right_fit, frame.shape)
        return annotate(result, curv, off)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sliding-window polynomial lane detection demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image file")
    group.add_argument("--video", help="Path to a video file (or 0 for webcam)")
    parser.add_argument(
        "--show-warped", action="store_true",
        help="Display the bird's-eye binary view alongside the main output")
    args = parser.parse_args()

    detector = SlidingWindowLaneDetector()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image: {args.image}")
        result = detector.process(frame)
        cv2.imshow("Sliding Window Lane Detection", result)
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
            cv2.imshow("Sliding Window Lane Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
