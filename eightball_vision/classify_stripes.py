import cv2
import numpy as np
from pathlib import Path

def classify_ball_type(ball_roi):
    """Return 'striped' or 'solid' based on edge density and brightness variation."""
    # Resize to standard size
    roi = cv2.resize(ball_roi, (80, 80))
    
    # --- Step 1: Preprocess ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Step 2: Edge detection ---
    edges = cv2.Canny(blur, 60, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # --- Step 3: Brightness variance (striped balls have white patches) ---
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    brightness_std = np.std(v)

    # --- Step 4: Decide based on thresholds ---
    if edge_density > 0.08 or brightness_std > 35:
        return "striped"
    else:
        return "solid"


def process_image(image_path, detections, output_path):
    """Take original image + bounding boxes and annotate each ball as solid/striped."""
    img = cv2.imread(str(image_path))
    annotated = img.copy()

    for det in detections:  # det = (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, det)
        ball_roi = img[y1:y2, x1:x2]
        if ball_roi.size == 0:
            continue

        ball_type = classify_ball_type(ball_roi)

        # Draw results
        color = (0, 255, 0) if ball_type == "solid" else (255, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, ball_type, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Example: Load one of your annotated results to reuse bounding boxes
    image_path = "data/photos/pool_table_16.jpg"

    # Example bounding boxes (replace this with your model’s output):
    # Each detection = (x1, y1, x2, y2)
    detections = [
        (200, 100, 260, 160),
        (300, 200, 360, 260),
        # ... from your Roboflow detections
    ]

    out_path = Path("output_classified.jpg")
    process_image(image_path, detections, out_path)
