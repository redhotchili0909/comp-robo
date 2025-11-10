"""
Interactive HSV Range Tuner for Pool Table Image
------------------------------------------------
Lets you visualize and adjust the HSV threshold range for blue felt detection.

Display:
- Left: Original image
- Right: Mask result (white = detected blue, black = non-blue)

Use the six sliders to tweak H, S, V lower/upper bounds interactively.
Press 's' to save the current mask visualization as 'hsv_mask_preview.jpg'.
Press 'q' to quit.
"""

import cv2
import numpy as np

# Load image
img = cv2.imread("data/photos/pool_table_16.jpg")
if img is None:
    raise FileNotFoundError("Could not load image.")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- initial HSV thresholds ---
lower = np.array([107, 126, 63])
upper = np.array([114, 255, 255])
# Create window
cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Tuner", 1200, 600)

# Create trackbars
def nothing(x): pass

cv2.createTrackbar("H lower", "HSV Tuner", lower[0], 179, nothing)
cv2.createTrackbar("S lower", "HSV Tuner", lower[1], 255, nothing)
cv2.createTrackbar("V lower", "HSV Tuner", lower[2], 255, nothing)
cv2.createTrackbar("H upper", "HSV Tuner", upper[0], 179, nothing)
cv2.createTrackbar("S upper", "HSV Tuner", upper[1], 255, nothing)
cv2.createTrackbar("V upper", "HSV Tuner", upper[2], 255, nothing)

print("Adjust sliders to tune HSV. Press 's' to save current mask. Press 'q' to quit.")

while True:
    # Read trackbar positions
    hL = cv2.getTrackbarPos("H lower", "HSV Tuner")
    sL = cv2.getTrackbarPos("S lower", "HSV Tuner")
    vL = cv2.getTrackbarPos("V lower", "HSV Tuner")
    hU = cv2.getTrackbarPos("H upper", "HSV Tuner")
    sU = cv2.getTrackbarPos("S upper", "HSV Tuner")
    vU = cv2.getTrackbarPos("V upper", "HSV Tuner")

    lower_blue = np.array([hL, sL, vL])
    upper_blue = np.array([hU, sU, vU])

    # Generate mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    non_blue = cv2.bitwise_not(mask)
    masked = cv2.bitwise_and(img, img, mask=non_blue)

    # Stack visualization
    mask_bgr = cv2.merge([mask, mask, mask])
    combined = np.hstack((img, mask_bgr, masked))

    cv2.imshow("HSV Tuner", combined)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("hsv_mask_preview.jpg", combined)
        print("Saved preview as hsv_mask_preview.jpg")

cv2.destroyAllWindows()
