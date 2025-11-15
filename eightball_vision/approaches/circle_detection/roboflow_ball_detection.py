"""
Pool Ball Detection using Roboflow ML Model
"""
from pathlib import Path
from dotenv import load_dotenv
from inference import get_model
import cv2
import os

load_dotenv()

OUTPUT_DIR = "results/ml_detection"

def save_image(filename, image):
    """Save image to the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath

def detect_balls(image_path):
    api_key = os.getenv("ROBOFLOW_API_KEY")
    # start Roboflow model
    model = get_model(model_id="balldetect-hefhg/1", api_key=api_key)
    # run inference
    result = model.infer(image_path, confidence=0.15, iou_threshold=0.3)[0]
    preds = result.predictions

    # read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read image")
        return
    
    # drop alpha channel if RGBA
    if (len(img.shape) == 3 and img.shape[2] == 4):
        bgr = img[:, :, :3]
    else:
        bgr = img

    # collect predictions in xyxy format
    xyxy = []
    confs = []
    classes = []
    for p in preds:
        x1 = p.x - p.width / 2
        y1 = p.y - p.height / 2
        x2 = p.x + p.width / 2
        y2 = p.y + p.height / 2
        xyxy.append([x1, y1, x2, y2])
        confs.append(float(p.confidence))
        classes.append(p.class_name)

    annotated = bgr.copy()

    # ensure output folders exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    img_name = Path(image_path).stem

    crops_subdir = f"{img_name}_balls"
    crops_dir_path = os.path.join(OUTPUT_DIR, crops_subdir)

    if not os.path.exists(crops_dir_path):
        os.makedirs(crops_dir_path)

    for i, (box, conf, cls) in enumerate(zip(xyxy, confs, classes), start=1):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(annotated.shape[1], x2), min(annotated.shape[0], y2)
        # draw box + label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        crop = annotated[y1:y2, x1:x2]
        if crop.size:
            crop_name = f"ball_{i}.png"
            # save individual crop
            save_image(f"{crops_subdir}/{crop_name}", crop)

    # save final annotated image
    save_image(f"{img_name}_annotated.png", annotated)


if __name__ == "__main__":
    detect_balls("../../data/photos/pool_table_0.png")
