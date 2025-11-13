"""
Pool Ball Detection using Color Segmentation Approach
"""
import os
import cv2
import numpy as np

OUTPUT_DIR = "results/color_segmentation"

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


def preprocess_lighting(bgr):
    """
    Normalize lighting conditions using CLAHE to handle uneven lighting and shadows.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    save_image("preprocessed_image.jpg", result)
    
    return result


def segment_balls_by_color(bgr):
    """
    Segment pool balls from the mat background using color-based segmentation.
    Automatically detects the dominant color and creates a mask for non-mat regions.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Extract histogram of hue channel to find dominant color (mat)
    hist = cv2.calcHist(hsv.astype(np.uint8), [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    
    # Define range for mat color
    hue_margin = 25
    lower_bound = np.array([max(dominant_hue - hue_margin, 0), 40, 40])
    upper_bound = np.array([min(dominant_hue + hue_margin, 179), 255, 255])
    mat_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Invert mask to get balls instead of mat
    balls_mask = cv2.bitwise_not(mat_mask)
    
    save_image("masked.jpg", balls_mask)
    
    return balls_mask


def find_circular_contours(mask):
    """
    Find circular ball regions using contour fitting.
    Filters contours by area, circularity, and radius.
    """

    min_area = 3000
    max_area = 20000
    min_circularity = 0.40
    min_radius = 20
    max_radius = 90

    # Create debug visualization
    debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    all_contours_img = debug_img.copy()
    filtered_img = debug_img.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours
    cv2.drawContours(all_contours_img, contours, -1, (0, 0, 255), 2)
    
    circles = []
    accepted_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by area
        if not (min_area < area < max_area):
            continue
        
        # Calculate circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter by circularity
        if circularity < min_circularity:
            continue

        # Get enclosing circle
        (x, y), r = cv2.minEnclosingCircle(cnt)
        
        # Filter by radius
        if r < min_radius or r > max_radius:
            continue
        accepted_contours.append(cnt)
            
        circles.append((int(x), int(y), int(r)))
    
    # Draw accepted contours and circles
    cv2.drawContours(filtered_img, accepted_contours, -1, (0, 255, 0), 2)
    for (x, y, r) in circles:
        cv2.circle(filtered_img, (x, y), int(r), (255, 0, 0), 2)
        cv2.circle(filtered_img, (x, y), 3, (255, 0, 0), -1)
    
    save_image("all_contours.jpg", all_contours_img)
    save_image("filtered_circles.jpg", filtered_img)
    
    return circles


def main(image_path):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load image
    img = load_image(image_path)
    save_image("original.jpg", img)
    
    # Preprocess lighting
    img = preprocess_lighting(img)
    
    # Segment balls by color
    mask = segment_balls_by_color(img)
    
    # Find circular contours
    circles = find_circular_contours(mask)

    return circles


if __name__ == "__main__":
    main("../../data/photos/pool_table_0.png")
