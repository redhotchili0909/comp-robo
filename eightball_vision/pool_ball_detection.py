"""
STEP 1 — Pool Ball Detection (Blue Table Segmentation)
------------------------------------------------------
Detects all pool balls by separating them from the blue table background
and drawing tight green circles around each detected ball.

Output:
    detected_balls_step1.jpg
"""

import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """Load image or raise a clear error if not found."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img

def segment_non_blue_balls(bgr: np.ndarray) -> np.ndarray:
    """Segment balls by removing the blue table background using adaptive thresholds."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # --- Step 1: Estimate average felt color automatically ---
    sample = hsv[::10, ::10, 0]  # sample every 10 pixels (for speed)
    hist = cv2.calcHist([sample.astype(np.uint8)], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))  # most frequent hue ≈ felt hue

    # --- Step 2: Build hue range around that felt color ---
    hue_margin = 15  # widen this if lighting is uneven
    lower_blue = np.array([max(dominant_hue - hue_margin, 0), 30, 40])
    upper_blue = np.array([min(dominant_hue + hue_margin, 179), 255, 255])

    # --- Step 3: Mask felt and invert to get balls ---
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    non_blue = cv2.bitwise_not(blue_mask)

    # --- Step 4: Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_OPEN, kernel, iterations=2)
    non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_CLOSE, kernel, iterations=3)
    non_blue = cv2.medianBlur(non_blue, 5)

    return non_blue

    return non_blue

def find_ball_circles(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Find circles (x, y, r) of likely pool balls using contours."""
    h, w = mask.shape[:2]
    image_area = h * w

    # Estimate reasonable area range for balls (relative to image size)
    min_area = image_area * 0.001  # ~0.1% of image
    max_area = image_area * 0.02   # ~2% of image

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        perimeter = max(cv2.arcLength(cnt, True), 1)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Real pool balls are very round — higher circularity threshold helps
        if circularity < 0.75:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 10 or r > 60:  # skip tiny or huge circles
            continue

        circles.append((int(x), int(y), int(r)))

    return circles

def preprocess_lighting(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalize brightness while preserving color."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def detect_balls(image_path: str):
    """Main entry: detect and save annotated output."""
    img = load_image(image_path)
    img = preprocess_lighting(img)
    mask = segment_non_blue_balls(img)
    circles = find_ball_circles(mask)

    # Draw circles
    output = img.copy()
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

    cv2.imwrite("detected_balls_step1.jpg", output)
    print(f"Detected {len(circles)} balls. Saved to detected_balls_step1.jpg")


if __name__ == "__main__":
    # Change path if needed
    detect_balls("data/photos/pool_table_1.jpg")
