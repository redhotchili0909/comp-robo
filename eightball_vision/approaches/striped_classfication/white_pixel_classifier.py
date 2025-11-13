"""
Stripe vs Solid classifier: white/black pixel percentages inside a ball mask.

Readable structure:
- Config thresholds at top
- Small helpers for I/O and masking
- Clear classify and overlay functions
"""

import os
import sys
import glob
import cv2
import numpy as np

# ----------------------------------
# Config
# ----------------------------------
# Output directory for annotated overlays
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "white_pixel")

# RGB thresholds
BLACK_RGB_MAX = 64
WHITE_RGB_MIN = 192

# Decision thresholds (percentages)
THRESH_BLACK_SOLID_PCT = 60.0
THRESH_WHITE_CUE_PCT = 60.0
THRESH_STRIPE_WHITE_MIN = 15.0
THRESH_STRIPE_WHITE_MAX = 50.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_detection = os.path.join(BASE_DIR, "circle_detection", "results", "ml_detection")
INPUT_PATH = os.path.join(ml_detection, "pool_table_0_balls")


def ensure_path(p):
    return p if p and (os.path.isdir(p) or os.path.isfile(p)) else None

def make_circle_mask(image_shape, center, radius):
    """Return a filled circle mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def compute_white_black_percentages(img, mask):
    """Compute white and black pixel percentages within mask using RGB thresholds."""
    total = cv2.countNonZero(mask)
    if total == 0:
        return 0.0, 0.0, total
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    is_black = (r < BLACK_RGB_MAX) & (g < BLACK_RGB_MAX) & (b < BLACK_RGB_MAX) & (mask == 255)
    is_white = (r > WHITE_RGB_MIN) & (g > WHITE_RGB_MIN) & (b > WHITE_RGB_MIN) & (mask == 255)
    black_pct = (np.sum(is_black) / float(total)) * 100.0
    white_pct = (np.sum(is_white) / float(total)) * 100.0
    return white_pct, black_pct, total


def classify_from_percentages(white_pct, black_pct):
    """Map percentages to a label and debug string."""
    dbg = f"W:{white_pct:.0f} B:{black_pct:.0f}"
    if black_pct > THRESH_BLACK_SOLID_PCT:
        return "Solid (8-Ball)", dbg
    if white_pct > THRESH_WHITE_CUE_PCT:
        return "Solid (Cue)", dbg
    if THRESH_STRIPE_WHITE_MIN < white_pct < THRESH_STRIPE_WHITE_MAX:
        return "Striped", dbg
    return "Solid", dbg


def classify_by_pixel_count(img, center, radius):
    """Main classifier: circle mask -> percentages -> decision."""
    mask = make_circle_mask(img.shape, center, radius)
    white_pct, black_pct, total = compute_white_black_percentages(img, mask)
    if total == 0:
        return "Unknown", "N/A"
    return classify_from_percentages(white_pct, black_pct)


def save_overlay(image, subdir, base_name, label, dbg, center, radius):
    # make sure output directory exists
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bgr = image.copy()
    h, w = bgr.shape[:2]

    # build circle mask
    mask = make_circle_mask(bgr.shape, center, radius)

    # compute white/black pixels like classifier for visualization
    r = bgr[:, :, 2]
    g = bgr[:, :, 1]
    b = bgr[:, :, 0]
    is_black = ((r < BLACK_RGB_MAX) & (g < BLACK_RGB_MAX) & (b < BLACK_RGB_MAX) & (mask == 255))
    is_white = ((r > WHITE_RGB_MIN) & (g > WHITE_RGB_MIN) & (b > WHITE_RGB_MIN) & (mask == 255))

    # color layers for overlays
    color_layer = np.zeros_like(bgr, dtype=np.uint8)
    color_layer[is_white] = (0, 255, 255)   # yellow tint for white-counted pixels
    color_layer[is_black] = (0, 0, 255)     # red tint for black-counted pixels

    alpha = np.zeros((h, w, 1), dtype=np.float32)
    alpha[is_white] = 0.35
    alpha[is_black] = 0.35

    blended = (bgr.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # draw circle outline
    cv2.circle(blended, center, radius, (0, 255, 0), 2)

    # put label and debug text
    text = f"{label}  [{dbg}]"
    cv2.putText(blended, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(blended, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(out_path, blended)
    return out_path


def get_image_list_and_subdir(path_in):
    """Return a list of image paths and a subdir name for outputs."""
    if os.path.isfile(path_in):
        return [path_in], (os.path.basename(os.path.dirname(path_in)) or "single")
    imgs = sorted(glob.glob(os.path.join(path_in, "*.png"))) + \
           sorted(glob.glob(os.path.join(path_in, "*.jpg")))
    return imgs, os.path.basename(os.path.normpath(path_in))


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
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        c = (w // 2, h // 2)
        r = int(0.6 * min(w, h))
        label, dbg = classify_by_pixel_count(img, c, r)
        if label.startswith("Striped"):
            stripes += 1
        elif label.startswith("Solid"):
            solids += 1
        print(f"{os.path.basename(p)} -> {label} [{dbg}]")

        # save annotated overlay next to results directory
        base = os.path.splitext(os.path.basename(p))[0]
        save_overlay(img, subdir, base, label, dbg, c, r)
    print(f"\nTotals: {stripes} stripes, {solids} solids")


if __name__ == "__main__":
    classify_path(INPUT_PATH)
