"""
Stripe vs Solid: white/black pixel percentages inside a ball mask
"""

import os
import glob
import cv2
import numpy as np

# output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "white_pixel")

# RGB thresholds
BLACK_RGB_MAX = 64
WHITE_RGB_MIN = 192

# percentage thresholds
THRESH_BLACK_SOLID_PCT = 60.0
THRESH_WHITE_CUE_PCT = 60.0
THRESH_STRIPE_WHITE_MIN = 15.0
THRESH_STRIPE_WHITE_MAX = 50.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_detection = os.path.join(BASE_DIR, "circle_detection", "results", "ml_detection")
INPUT_PATH = os.path.join(ml_detection, "pool_table_0_balls")

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
    """Determine ball type using set threshold"""
    percentage = f"W:{white_pct} B:{black_pct}"
    if black_pct > THRESH_BLACK_SOLID_PCT:
        return "Solid (8-Ball)", percentage
    if white_pct > THRESH_WHITE_CUE_PCT:
        return "Solid (Cue)", percentage
    if THRESH_STRIPE_WHITE_MIN < white_pct < THRESH_STRIPE_WHITE_MAX:
        return "Striped", percentage
    return "Solid", percentage


def classify_by_pixel_count(img, center, radius):
    """Main classifier: circle mask -> percentages -> decision."""
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    white_pct, black_pct, total = compute_white_black_percentages(img, mask)
    if total == 0:
        return "Unknown", "0"
    return classify_from_percentages(white_pct, black_pct)


def save_overlay(image, subdir, base_name, label, percentage, center, radius):

    out_dir = os.path.join(OUTPUT_DIR, subdir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bgr = image.copy()

    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)


    # draw circle outline
    cv2.circle(bgr, center, radius, (0, 255, 0), 2)

    # put label and debug text
    text = f"{label}  [{percentage}]"
    cv2.putText(bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(out_path, bgr)
    return out_path



def main(path_in):
    
    imgs = sorted(glob.glob(os.path.join(path_in, "*.png"))) + \
           sorted(glob.glob(os.path.join(path_in, "*.jpg")))
    
    if not imgs:
        return
    
    subdir = os.path.basename(os.path.normpath(path_in))
    stripes = solids = 0

    for p in imgs:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        c = (w // 2, h // 2)
        r = int(0.6 * min(w, h))
        label, percentage = classify_by_pixel_count(img, c, r)
        if label.startswith("Striped"):
            stripes += 1
        elif label.startswith("Solid"):
            solids += 1
        print(f"{os.path.basename(p)} -> {label} [{percentage}]")

        base = os.path.splitext(os.path.basename(p))[0]
        save_overlay(img, subdir, base, label, percentage, c, r)

if __name__ == "__main__":
    main(INPUT_PATH)
