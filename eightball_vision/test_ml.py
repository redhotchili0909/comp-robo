from pathlib import Path
from dotenv import load_dotenv
from inference import get_model
import supervision as sv
import cv2
import numpy as np
from classify_stripes import classify_ball_type

# Load environment variables
load_dotenv()

import os
api_key = os.getenv("ROBOFLOW_API_KEY")
model = get_model(model_id="balldetect-hefhg/1", api_key=api_key)


here = Path(__file__).parent
photos_dir = here / "data/photos"
out_dir = here / "output_images"
out_dir.mkdir(parents=True, exist_ok=True)

# Get all image files
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
image_files = [f for f in photos_dir.iterdir() if f.suffix.lower() in image_extensions]
image_files.sort()

print(f"Found {len(image_files)} images to process")

for idx, image_path in enumerate(image_files, 1):
    print(f"Processing [{idx}/{len(image_files)}]: {image_path.name}", end=" ... ")
    try:
        result = model.infer(str(image_path), confidence=0.15, iou_threshold=0.3)[0]
        preds = result.predictions

        xyxy, confidences, class_ids = [], [], []
        class_name_to_id, class_names = {}, []

        for p in preds:
            xyxy.append([p.x - p.width/2, p.y - p.height/2, p.x + p.width/2, p.y + p.height/2])
            confidences.append(float(p.confidence))
            cname = p.class_name
            if cname not in class_name_to_id:
                class_name_to_id[cname] = len(class_name_to_id)
                class_names.append(cname)
            class_ids.append(class_name_to_id[cname])

        detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=float),
            confidence=np.array(confidences, dtype=float),
            class_id=np.array(class_ids, dtype=int),
        )

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"⚠ Failed to read image: {image_path.name}")
            continue

        has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
        bgr_image = image[:, :, :3] if has_alpha else image

        annotated = sv.BoxAnnotator(thickness=2).annotate(scene=bgr_image.copy(), detections=detections)
        annotated = sv.LabelAnnotator(text_thickness=1, text_scale=0.5).annotate(scene=annotated, detections=detections)
        balls_dir = out_dir / f"{image_path.stem}_balls"
        balls_dir.mkdir(parents=True, exist_ok=True)

        for j, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            ball_roi = bgr_image[y1:y2, x1:x2]

            if ball_roi.size == 0:
                continue

            ball_path = balls_dir / f"ball_{j+1}.png"
            cv2.imwrite(str(ball_path), ball_roi)

            ball_type = classify_ball_type(ball_roi)
            color = (0, 255, 0) if ball_type == "solid" else (255, 0, 255)
            cv2.putText(annotated, ball_type, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"  → Ball at ({x1},{y1}) is {ball_type}")

        out_path = out_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  ✓ Detected {len(detections)} objects → {out_path.name}")

    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {e}")
        continue


print(f"\n✓ Done! Processed {len(image_files)} images")
print(f"Output saved to: {out_dir}")