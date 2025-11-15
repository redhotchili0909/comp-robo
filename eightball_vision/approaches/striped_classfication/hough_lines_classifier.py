"""
Stripe vs Solid classifier: counts straight lines inside a circular mask using Hough Lines
"""

import os
import sys
import glob
import cv2
import numpy as np

# output directory for annotated overlays
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "hough_lines")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_detection = os.path.join(BASE_DIR, "circle_detection", "results", "ml_detection")
INPUT_PATH = os.path.join(ml_detection, "pool_table_0_balls")

CANNY_LOW = 60
CANNY_HIGH = 150
HOUGH_THRESH = 10
HOUGH_MIN_LINE_FRAC = 0.4  # as a fraction of radius
HOUGH_MAX_GAP = 5

def classify_by_lines(canny_img, center, radius):
    mask = np.zeros(canny_img.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(canny_img, canny_img, mask=mask)
    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=int(radius * HOUGH_MIN_LINE_FRAC),
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is not None:
        return "Striped", f"Lines:{len(lines)}", lines
    else:
        return "Solid", "Lines:0", lines


def save_overlay(bgr, center, radius, lines, label, dbg, subdir, base_name):
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    vis = bgr.copy()
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.circle(vis, center, radius, (0, 255, 0), 2)
    text = f"{label}  [{dbg}]"
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(out_path, vis)
    return out_path


def main(path_in):

    imgs = sorted(glob.glob(os.path.join(path_in, "*.png"))) + \
            sorted(glob.glob(os.path.join(path_in, "*.jpg")))

    stripes = solids = 0
    for p in imgs:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        c = (w // 2, h // 2)
        r = int(0.45 * min(w, h))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        label, dbg, lines = classify_by_lines(canny, c, r)
        if label.startswith("Striped"):
            stripes += 1
        elif label.startswith("Solid"):
            solids += 1

        base = os.path.splitext(os.path.basename(p))[0]
        subdir = os.path.basename(os.path.dirname(p)) or "single"
        save_overlay(bgr, c, r, lines, label, dbg, subdir, base)


if __name__ == "__main__":
    main(INPUT_PATH)
