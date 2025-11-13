"""
Stripe vs Solid classifier: hue standard deviation inside a circle mask.

Readable structure with small helpers and clear constants; saves an overlay per crop.
"""

import os
import sys
import glob
import cv2
import numpy as np

# output directory for annotated overlays
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "contrast_boundary")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_detection = os.path.join(BASE_DIR, "circle_detection", "results", "ml_detection")
INPUT_PATH = os.path.join(ml_detection, "pool_table_0_balls")

# thresholds
BLACK_V_MAX = 40
WHITE_S_MAX = 30
WHITE_V_MIN = 180
HUE_STD_STRIPED = 20.0


def ensure_path(p):
    return p if p and (os.path.isdir(p) or os.path.isfile(p)) else None


def classify_by_hue_std(hsv_img, center, radius):
    mask = np.zeros(hsv_img.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    total = cv2.countNonZero(mask)
    if total == 0:
        return "Unknown", "N/A"
    h = hsv_img[:, :, 0][mask == 255]
    s = hsv_img[:, :, 1][mask == 255]
    v = hsv_img[:, :, 2][mask == 255]
    black_pct = np.sum(v < BLACK_V_MAX) / total
    white_pct = np.sum((s < WHITE_S_MAX) & (v > WHITE_V_MIN)) / total
    if black_pct > 0.60:
        return "Solid (8-Ball)", "8"
    if white_pct > 0.60:
        return "Solid (Cue)", "Cue"
    hue_std = float(np.std(h))
    dbg = f"HueStd:{hue_std:.1f}"
    return ("Striped", dbg) if hue_std > HUE_STD_STRIPED else ("Solid", dbg)


def _class_to_color(label):
    # pick a tint per class for visualization (BGR)
    if label.startswith("Solid (8-Ball)"):
        return (0, 0, 255)   # red
    if label.startswith("Solid (Cue)"):
        return (255, 0, 0)   # blue
    if label.startswith("Striped"):
        return (0, 255, 255) # yellow
    if label.startswith("Solid"):
        return (0, 255, 0)   # green
    return (200, 200, 200)


def save_overlay(bgr, center, radius, label, dbg, subdir, base_name):
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    color = _class_to_color(label)
    tint = np.zeros_like(bgr)
    tint[mask == 255] = color

    alpha = np.zeros((h, w, 1), dtype=np.float32)
    alpha[mask == 255] = 0.25
    blended = (bgr.astype(np.float32) * (1.0 - alpha) + tint.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.circle(blended, center, radius, (0, 255, 0), 2)
    text = f"{label}  [{dbg}]"
    cv2.putText(blended, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(blended, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(out_path, blended)
    return out_path


def classify_path(path_in):
    if os.path.isfile(path_in):
        imgs = [path_in]
        print(f"File: {path_in}\n")
        subdir = os.path.basename(os.path.dirname(path_in)) or "single"
    else:
        imgs = sorted(glob.glob(os.path.join(path_in, "*.png"))) + \
               sorted(glob.glob(os.path.join(path_in, "*.jpg")))
        print(f"Folder: {path_in}")
        print(f"Found {len(imgs)} crops\n")
        if not imgs:
            return
        subdir = os.path.basename(os.path.normpath(path_in))
    stripes = solids = 0
    for p in imgs:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        c = (w // 2, h // 2)
        r = int(0.45 * min(w, h))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        label, dbg = classify_by_hue_std(hsv, c, r)
        if label.startswith("Striped"):
            stripes += 1
        elif label.startswith("Solid"):
            solids += 1
        print(f"{os.path.basename(p)} -> {label} [{dbg}]")

        base = os.path.splitext(os.path.basename(p))[0]
        save_overlay(bgr, c, r, label, dbg, subdir, base)
    print(f"\nTotals: {stripes} stripes, {solids} solids")


if __name__ == "__main__":
    classify_path(INPUT_PATH)
