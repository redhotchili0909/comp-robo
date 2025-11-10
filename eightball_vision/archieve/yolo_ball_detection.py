import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model("data/photos-jpg/pool_table_16.jpg", imgsz=1280, conf=0.25)

img = cv2.imread("data/photos-jpg/pool_table_16.jpg")
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    w, h = x2 - x1, y2 - y1
    aspect = w / float(h)
    circularity = min(aspect, 1/aspect)

    if circularity > 0.1:  # close to square = roughly circular
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("filtered_detections.jpg", img)
