"""
Pool Ball Detection using Hough Circle Transform Approach
"""
import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "hough_circle")

def save_image(filename, image):
    """Save image to the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath


def load_image(image_path):
    """Load an image from the specified path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def mask_mat(bgr):
    """
    Create a mask to isolate balls from the mat background.
    Detects the dominant mat color and masks it out.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # extract histogram of hue channel to find dominant color (mat)
    hist = cv2.calcHist(hsv.astype(np.uint8), [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    
    # define range for mat color
    hue_margin = 25
    lower_bound = np.array([max(dominant_hue - hue_margin, 0), 40, 40])
    upper_bound = np.array([min(dominant_hue + hue_margin, 179), 255, 255])
    mat_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # invert mask to get balls instead of mat
    balls_mask = cv2.bitwise_not(mat_mask)

    return balls_mask


def preprocess_for_hough(bgr):
    """
    Preprocess image for Hough Circle Transform.
    """
    
    # convert to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # apply mask
    gray = cv2.bitwise_and(gray, gray)
    
    # apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    return blurred


def detect_with_hough_circles(gray, bgr):
    """
    Detect circles using Hough Circle Transform.
    """
    
    # hough Circle parameters
    dp = 1.2
    min_dist = 38
    param1 = 100
    param2 = 30
    min_radius = 50
    max_radius = 80

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    detected_circles = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # create visualization
        vis_img = bgr.copy()
        
        for i, (x, y, r) in enumerate(circles[0]):
            detected_circles.append((int(x), int(y), int(r)))
            # draw on visualization
            cv2.circle(vis_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis_img, (x, y), 2, (0, 0, 255), 3)
        
        filepath = save_image("detected_circles.jpg", vis_img)
    else:
        print("  No circles detected")
    
    return detected_circles

def main(image_path):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # load image
    img = load_image(image_path)
    save_image("original.jpg", img)
    
    # mask out the mat to focus on balls
    mat_mask = mask_mat(img)
    
    # preprocess for Hough with mask
    gray = preprocess_for_hough(img)

    # detect circles with Hough Transform
    circles = detect_with_hough_circles(gray, img)
    
    return circles


if __name__ == "__main__":
    default_image = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "..", "data", "photos", "pool_table_0.png")
    )
    main(default_image)
