import cv2
import numpy as np
import os

# --- User configuration ---
num_balls_expected = 15   # set manually

def load_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def segment_non_blue_balls(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sample = hsv[::7, ::7, 0]
    hist = cv2.calcHist([sample.astype(np.uint8)], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    hue_margin = 12
    lower_felt = np.array([max(dominant_hue - hue_margin, 0), 40, 40])
    upper_felt = np.array([min(dominant_hue + hue_margin, 179), 255, 255])
    felt_mask = cv2.inRange(hsv, lower_felt, upper_felt)
    cv2.imwrite("debug_felt_mask.jpg", felt_mask)
    return felt_mask, dominant_hue


def find_ball_circles(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape[:2]
    image_area = h * w
    min_area = image_area * 0.005
    max_area = image_area * 0.09

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        perimeter = max(cv2.arcLength(cnt, True), 1)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.26:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 5 or r > 90:
            continue
        circles.append((int(x), int(y), int(r)))
    return circles


def preprocess_lighting(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def recover_dark_balls_with_hough(img_bgr: np.ndarray, dominant_hue: int,
                                  existing: list[tuple[int,int,int]]) -> list[tuple[int,int,int]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    dark = cv2.inRange(v, 0, 80)
    felt = cv2.inRange(hsv,
                       (max(dominant_hue - 12, 0), 30, 40),
                       (min(dominant_hue + 12, 179), 255, 255))
    roi = cv2.bitwise_and(dark, cv2.bitwise_not(felt))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    g = gray.copy()
    g[roi == 0] = 0
    g = cv2.medianBlur(g, 5)

    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=38,
        param1=80, param2=14,
        minRadius=12, maxRadius=40
    )

    extra = []
    if circles is not None:
        for (x, y, r) in np.uint16(np.around(circles))[0]:
            keep = True
            for (x0, y0, r0) in existing:
                if (x - x0)**2 + (y - y0)**2 < (0.6 * max(r, r0))**2:
                    keep = False
                    break
            if keep:
                extra.append((int(x), int(y), int(r)))

    cv2.imwrite("debug_dark_roi.jpg", roi)
    return extra


def detect_balls(image_path: str):
    img = load_image(image_path)
    img = preprocess_lighting(img)
    mask, dominant_hue = segment_non_blue_balls(img)
    cv2.imwrite("detected_balls_step1_mask.jpg", mask)

    # label origin = "segment"
    circles_segment = [(x, y, r, "segment") for (x, y, r) in find_ball_circles(mask)]
    circles_dark = [(x, y, r, "dark_hough") for (x, y, r) in recover_dark_balls_with_hough(img, dominant_hue, circles_segment)]

    circles = circles_segment + circles_dark

    output = img.copy()
    for (x, y, r, src) in circles:
        color = (0, 255, 0) if src == "segment" else (0, 165, 255)
        cv2.circle(output, (x, y), r, color, 2)
        cv2.putText(output, src, (x - r, y - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.imwrite("detected_balls_step1.jpg", output)
    print(f"Detected {len(circles)} balls (expected {num_balls_expected}).")
    print(f"Segment: {len(circles_segment)}, Dark-Hough: {len(circles_dark)}")


if __name__ == "__main__":
    detect_balls("data/photos/pool_table_14.png")
