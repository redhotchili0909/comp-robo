from pathlib import Path
from dotenv import load_dotenv
from inference import get_model
import supervision as sv
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Load model locally
model = get_model(model_id="balldetect-hefhg/1")

here = Path(__file__).parent
photos_dir = here / "data/photos"
out_dir = here / "output_images"
out_dir.mkdir(parents=True, exist_ok=True)

# Get all image files
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
image_files = [f for f in photos_dir.iterdir() if f.suffix.lower() in image_extensions]
image_files.sort()

print(f"Found {len(image_files)} images to process")

# Process each image
for idx, image_path in enumerate(image_files, 1):
    print(f"Processing [{idx}/{len(image_files)}]: {image_path.name}", end=" ... ")
    try:
        result = model.infer(str(image_path), confidence=0.15, iou_threshold=0.3)[0]
        
        # Convert inference result to supervision Detections
        preds = result.predictions
        xyxy = []
        confidences = []
        class_ids = []
        class_name_to_id = {}
        class_names = []
        
        for p in preds:
            # Inference SDK returns xyxy directly
            xyxy.append([p.x - p.width/2, p.y - p.height/2, p.x + p.width/2, p.y + p.height/2])
            confidences.append(float(p.confidence))
            cname = p.class_name
            if cname not in class_name_to_id:
                class_name_to_id[cname] = len(class_name_to_id)
                class_names.append(cname)
            class_ids.append(class_name_to_id[cname])
        
        detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=float) if xyxy else np.zeros((0, 4), dtype=float),
            confidence=np.array(confidences, dtype=float) if confidences else np.array([], dtype=float),
            class_id=np.array(class_ids, dtype=int) if class_ids else np.array([], dtype=int),
        )
        
        # Read image with alpha channel (transparency) if present
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  ⚠ Failed to read image: {image_path.name}")
            continue
        
        # Check if image has alpha channel
        has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
        
        # Filter out detections in transparent areas (if image has alpha channel)
        if has_alpha and len(detections) > 0:
            alpha_channel = image[:, :, 3]
            valid_mask = []
            valid_class_names_filtered = []
            
            for idx, bbox in enumerate(detections.xyxy):
                # Get center point of bounding box
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Check if center is within image bounds and on non-transparent pixel
                if (0 <= center_y < alpha_channel.shape[0] and 
                    0 <= center_x < alpha_channel.shape[1] and
                    alpha_channel[center_y, center_x] > 0):  # Alpha > 0 means not transparent
                    valid_mask.append(True)
                    valid_class_names_filtered.append(class_names[detections.class_id[idx]])
                else:
                    valid_mask.append(False)
            
            # Filter detections
            valid_mask = np.array(valid_mask)
            if valid_mask.any():
                detections = sv.Detections(
                    xyxy=detections.xyxy[valid_mask],
                    confidence=detections.confidence[valid_mask],
                    class_id=detections.class_id[valid_mask],
                )
                class_names = valid_class_names_filtered
            else:
                # No valid detections
                detections = sv.Detections.empty()
                class_names = []
        
        # Pretty labels with class name and confidence
        labels = [
            f"{class_names[idx]} {conf:.2f}"
            for idx, conf in enumerate(detections.confidence)
        ]
        
        if has_alpha:
            # Extract alpha channel and RGB channels
            alpha_channel = image[:, :, 3]
            bgr_image = image[:, :, :3]
        else:
            bgr_image = image
        
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        
        annotated = box_annotator.annotate(scene=bgr_image.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    
        # Save output with same filename + _annotated
        out_path = out_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  ✓ Detected {len(detections)} objects → {out_path.name}")
        
    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {e}")
        continue

print(f"\n✓ Done! Processed {len(image_files)} images")
print(f"Output saved to: {out_dir}")