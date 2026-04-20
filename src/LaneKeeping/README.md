# Lane Keeping Demo – Lane Detection Algorithms

This sub-project shows three progressively more capable approaches to finding
lane lines in a front-facing camera image, starting from the simplest possible
method and working up to the industry-standard sliding-window + polynomial-fit
pipeline.

---

## Files

| File | Description |
|------|-------------|
| `lane_detection_basic.py` | Baseline: Canny edges + Hough lines + slope filtering |
| `lane_detection_hough_improved.py` | Improved Hough: HLS colour filtering + robust line fitting + temporal smoothing |
| `lane_detection_sliding_window.py` | Advanced: Bird's-eye warp + sliding window + polynomial curve fitting |
| `test_lane_detection.py` | Pytest unit tests for all three modules |

---

## Quick Start

```bash
pip install opencv-python numpy pytest

# Single image
python lane_detection_basic.py            --image road.jpg
python lane_detection_hough_improved.py   --image road.jpg
python lane_detection_sliding_window.py   --image road.jpg

# Video / webcam
python lane_detection_basic.py            --video road.mp4
python lane_detection_hough_improved.py   --video 0        # webcam
python lane_detection_sliding_window.py   --video road.mp4

# Tests
pytest test_lane_detection.py -v
```

---

## Algorithm Comparison

### 1. Basic: Canny + Hough + Slope Filtering (`lane_detection_basic.py`)

```
BGR frame
  └─► Grayscale
        └─► Gaussian blur
              └─► Canny edge detection
                    └─► Fixed trapezoidal ROI mask
                          └─► Hough line segments
                                └─► Slope-sign split (left / right)
                                      └─► Average all segments per side
                                            └─► Draw two straight lines
```

**Pros:** Very fast, easy to understand, no dependencies beyond OpenCV.  
**Cons:**
- Only fits straight lines – fails on curved roads.
- Sensitive to shadows and lighting changes because no colour filtering.
- Frame-to-frame results can flicker badly.

---

### 2. Improved Hough (`lane_detection_hough_improved.py`)

Adds three improvements on top of the basic approach:

1. **HLS colour masking** – isolates white and yellow lane markings in HLS
   space *before* edge detection.  This dramatically reduces the noise from
   the road surface and shadow regions that confuse the basic method.

2. **Least-squares line fitting** – rather than averaging all Hough segment
   parameters, the endpoints of every accepted segment are collected and a
   single robust least-squares polynomial (degree 1) is fitted.  Outlier
   segments have far less influence.

3. **Exponential moving average (EMA) temporal smoother** – each lane line
   position is smoothed across recent frames using an EMA.  This eliminates
   flickering and keeps the visualisation stable during brief detection gaps.

```
BGR frame
  └─► HLS white/yellow mask
        └─► Grayscale → Gaussian blur → Canny
              └─► Dynamic trapezoidal ROI
                    └─► Hough line segments
                          └─► Slope filter + cluster (left / right)
                                └─► Least-squares line fit per side
                                      └─► EMA temporal smoother
                                            └─► Draw lines + lane fill
```

**Pros:** Handles shadows and yellow markings, smoother output.  
**Cons:** Still a straight-line model – curved roads remain a problem.

---

### 3. Sliding Window + Polynomial Curve Fitting (`lane_detection_sliding_window.py`)

This is the approach used in the Udacity Self-Driving Car Nanodegree and
widely adopted in research prototypes:

1. **Perspective warp (bird's-eye view)** – a homography matrix transforms
   the front-facing view into a top-down view.  Parallel lane lines that
   appear to converge in perspective now appear as nearly vertical stripes,
   which are far easier to fit with a polynomial.

2. **Multi-channel binary thresholding** – three independent thresholds are
   combined:
   - *Sobel-x on the L channel* – detects vertical edges.
   - *S channel (HLS)* – robust to illumination for both white and yellow.
   - *B channel (LAB)* – especially sensitive to yellow markings.

3. **Histogram peak detection** – summing pixel values vertically over the
   lower half of the warped binary image gives a histogram whose two dominant
   peaks are the starting x-positions of the left and right lanes.

4. **Sliding window search** – starting from the histogram peaks, a fixed
   rectangular window slides upward through the image, re-centering itself on
   the mean x-position of active pixels in each step.  All captured pixels
   are the lane pixel candidates.

5. **2nd-degree polynomial fit** – `x = Ay² + By + C` is fitted to the
   captured pixels via `numpy.polyfit`.  A polynomial correctly represents
   curved roads that a straight line cannot model.

6. **Targeted search** – once a good polynomial fit is available, subsequent
   frames search only in a narrow margin around that polynomial.  This is
   much faster than running the full sliding-window search every frame and
   handles transient detection failures gracefully.

7. **Curvature and lateral offset** – the radius of curvature (metres) and
   the vehicle's lateral offset from the lane centre (metres) are computed
   from the fitted polynomials using standard differential-geometry formulae.

8. **Polynomial smoothing** – coefficients are averaged over several recent
   frames to stabilise the output.

```
BGR frame
  └─► Perspective warp (bird's-eye)
        └─► Multi-channel binary threshold
              └─► Histogram peak detection
                    └─► Sliding window OR targeted search
                          └─► numpy.polyfit (degree 2)
                                └─► Polynomial smoothing
                                      └─► Unwarp lane overlay
                                            └─► Curvature + offset annotation
```

**Pros:**
- Correctly models curved roads.
- Pixel-level evidence – robust to partial occlusion.
- Provides physically meaningful curvature and lateral offset.
- Efficient video processing via targeted search.

**Cons:**
- Requires perspective-warp calibration (tuning `SRC_POINTS` per camera rig).
- Fails on very sharp bends outside the bird's-eye field of view.
- Still image-processing-based – struggles at night or in heavy rain without
  additional preprocessing.

---

## Further Reading and Next Steps

| Technique | Description |
|-----------|-------------|
| **RANSAC-based line / curve fitting** | Robust to outliers by random sampling; more reliable than least-squares when noise is high. |
| **Kalman filter lane tracking** | A proper state-space smoother; predicts lane position based on vehicle speed/yaw, handles long occlusions. |
| **Inverse Perspective Mapping (IPM) calibration** | Use a checker-board or known road markings to compute the exact warp matrix for your specific camera. |
| **Deep learning (CNN segmentation)** | Models such as SCNN, LaneNet, Ultra-Fast Lane Detection, or CLRNet learn to segment lane pixels end-to-end and can handle complex scenarios (night, rain, faded markings, complex intersections) that rule-based methods cannot. |
| **Anchor-based lane detection** | Row-anchor approaches (e.g. Ultra-Fast Lane Detection) predict lane positions at each row of the image in a single forward pass, achieving real-time performance even on embedded hardware. |
| **Transformer-based detection** | LSTR and similar models apply attention to capture global context, excelling at lane intersections and lane changes. |

For a production lane-keeping system, the sliding-window polynomial pipeline
is a solid starting point.  Pair it with a Kalman filter for tracking and
consider replacing or augmenting it with a CNN segmentation model for
robustness in adverse conditions.
