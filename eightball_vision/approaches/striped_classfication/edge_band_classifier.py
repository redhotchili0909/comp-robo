"""
Stripe vs Solid classifier using internal edge density inside a non-mat ROI.

Steps:
- Build a non-mat mask by detecting the dominant mat hue and inverting it
- Optionally erode the ROI to focus on interior
- Compute Canny edges and the edge density within the ROI
- Quick guards for cue and 8-ball based on HSV stats

Outputs:
- Prints a label per image and totals
- Saves an overlay per image to results/edge_band/<subdir>
"""

import os
import glob
import cv2
import numpy as np

# ----------------------------------
# Paths and config
# ----------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "edge_band")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DET_DIR = os.path.join(BASE_DIR, "circle_detection", "results", "ml_detection")
INPUT_PATH = os.path.join(ML_DET_DIR, "pool_table_0_balls")

# Edge + decision thresholds
EDGE_LOW = 60
EDGE_HIGH = 150
EDGE_DENSITY_STRIPED = 0.10

# HSV thresholds for quick cue/8 checks
BLACK_V_MAX = 40
WHITE_S_MAX = 30
WHITE_V_MIN = 180


# ----------------------------------
# Helpers
# ----------------------------------
def get_image_list_and_subdir(path_in: str):
    if os.path.isfile(path_in):
        return [path_in], (os.path.basename(os.path.dirname(path_in)) or "single")
    imgs = sorted(glob.glob(os.path.join(path_in, "*.png"))) + \
           sorted(glob.glob(os.path.join(path_in, "*.jpg")))
    return imgs, os.path.basename(os.path.normpath(path_in))


def compute_non_mat_mask(bgr_img):
    """Return (non_mat_mask, center, radius) for visualization.

    We detect the dominant hue (mat), threshold a band around it, then invert.
    The largest remaining blob is kept as the ROI.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # estimate dominant mat hue from valid pixels
    valid = (s > 40) & (v > 40)
    if np.count_nonzero(valid) < 50:
        mask = np.ones(h.shape, dtype=np.uint8) * 255
    else:
        h_valid = h[valid].astype(np.uint8)
        hist = cv2.calcHist([h_valid], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hist))
        hue_margin = 20
        lower = np.array([max(dominant_hue - hue_margin, 0), 40, 40])
        upper = np.array([min(dominant_hue + hue_margin, 179), 255, 255])
        mat_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mat_mask)

    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # keep largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h_img, w_img = bgr_img.shape[:2]
        return mask, (w_img // 2, h_img // 2), int(0.45 * min(w_img, h_img))
    largest = max(contours, key=cv2.contourArea)
    refined = np.zeros_like(mask)
    cv2.drawContours(refined, [largest], -1, 255, thickness=-1)
    (x, y), r = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(r)
    return refined, center, radius


def classify_by_edge_density(hsv_img, canny_img, roi_mask):
    total = cv2.countNonZero(roi_mask)
    if total == 0:
        return "Unknown", "N/A"

    # quick cue / 8-ball checks on full ROI
    v = hsv_img[:, :, 2][roi_mask == 255]
    s = hsv_img[:, :, 1][roi_mask == 255]
    black_pct = np.sum(v < BLACK_V_MAX) / float(total)
    white_pct = np.sum((s < WHITE_S_MAX) & (v > WHITE_V_MIN)) / float(total)
    if black_pct > 0.60:
        return "Solid (8-Ball)", "8"
    if white_pct > 0.60:
        return "Solid (Cue)", "Cue"

    # erode ROI slightly, count edges inside
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_eroded = cv2.erode(roi_mask, kernel, iterations=1)
    total_eroded = cv2.countNonZero(roi_eroded)
    if total_eroded == 0:
        roi_eroded = roi_mask
        total_eroded = total

    internal_edges = cv2.bitwise_and(canny_img, canny_img, mask=roi_eroded)
    edge_density = cv2.countNonZero(internal_edges) / float(total_eroded)
    dbg = f"Edge:{edge_density:.2f}"
    return ("Striped", dbg) if edge_density > EDGE_DENSITY_STRIPED else ("Solid", dbg)


def save_overlay(bgr, canny, roi_mask, center, radius, label, dbg, subdir, base_name):
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    h, w = bgr.shape[:2]
    vis = bgr.copy()

    # tint ROI lightly in green
    tint = np.zeros_like(vis)
    tint[roi_mask == 255] = (0, 255, 0)
    alpha_roi = np.zeros((h, w, 1), dtype=np.float32)
    alpha_roi[roi_mask == 255] = 0.15
    vis = (vis.astype(np.float32) * (1.0 - alpha_roi) + tint.astype(np.float32) * alpha_roi)
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # edges in red within ROI
    internal_edges = cv2.bitwise_and(canny, canny, mask=roi_mask)
    red = np.zeros_like(vis)
    red[internal_edges > 0] = (0, 0, 255)
    alpha_edges = (internal_edges > 0).astype(np.float32)[..., None] * 0.6
    vis = (vis.astype(np.float32) * (1.0 - alpha_edges) + red.astype(np.float32) * alpha_edges)
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # reference circle and text
    cv2.circle(vis, center, radius, (0, 255, 0), 2)
    text = f"{label}  [{dbg}]"
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(out_path, vis)
    return out_path


# ----------------------------------
# Main
# ----------------------------------
def classify_path(path_in):
    imgs, subdir = get_image_list_and_subdir(path_in)
    if not imgs:
        print(f"Found 0 crops in {path_in}")
        return
    if os.path.isfile(path_in):
        print(f"File: {path_in}\n")
    else:
        print(f"Folder: {path_in}")
        print(f"Found {len(imgs)} crops\n")

    stripes = solids = 0
    for p in imgs:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        roi_mask, center, radius = compute_non_mat_mask(bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, EDGE_LOW, EDGE_HIGH)

        label, dbg = classify_by_edge_density(hsv, canny, roi_mask)
        if label.startswith("Striped"):
            stripes += 1
        elif label.startswith("Solid"):
            solids += 1
        print(f"{os.path.basename(p)} -> {label} [{dbg}]")

        base = os.path.splitext(os.path.basename(p))[0]
        save_overlay(bgr, canny, roi_mask, center, radius, label, dbg, subdir, base)

    print(f"\nTotals: {stripes} stripes, {solids} solids")


if __name__ == "__main__":
    classify_path(INPUT_PATH)
 
