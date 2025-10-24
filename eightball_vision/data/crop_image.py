import cv2
import numpy as np
import os
import glob

def isolate_table_keep_balls(
    image_path,
    output_path,
    rectify=False,
):
    """
    Keep EVERYTHING on the felt (balls, chalk, reflections), remove outside.
    Saves a PNG with transparency outside the table.
    Also saves a contour overlay and binary mask for sanity checking.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    H, W = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 30, 20], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No table-sized blue contour found. Try widening HSV ranges.")
        return
    table_ct = max(contours, key=cv2.contourArea)
    if cv2.contourArea(table_ct) < 0.05 * H * W:
        print("Largest blue region too small to be the table; adjust thresholds.")
        return

    peri = cv2.arcLength(table_ct, True)
    approx = cv2.approxPolyDP(table_ct, 0.01 * peri, True)

    root, ext = os.path.splitext(output_path)

    table_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(table_mask, [table_ct], -1, 255, thickness=cv2.FILLED)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, table_mask])

    x, y, w, h = cv2.boundingRect(table_ct)
    cropped = rgba[y:y+h, x:x+w]

    if not rectify:
        cv2.imwrite(output_path, cropped)
        print(f"Saved table-isolated image (keeps balls) to {output_path}")
        return

    # ectify to top-down if we have 4 corners
    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(np.float32)

        # Order corners: top-left, top-right, bottom-right, bottom-left
        idx = np.argsort(pts[:,1])
        top = pts[idx[:2]][np.argsort(pts[idx[:2],0])]
        bottom = pts[idx[2:]][np.argsort(pts[idx[2:],0])]
        tl, tr = top
        bl, br = bottom

        width_top = np.linalg.norm(tr - tl)
        width_bot = np.linalg.norm(br - bl)
        height_l  = np.linalg.norm(bl - tl)
        height_r  = np.linalg.norm(br - tr)
        W_out = int(max(width_top, width_bot))
        H_out = int(max(height_l, height_r))
        W_out = max(W_out, 10)
        H_out = max(H_out, 10)

        dst = np.array([[0,0],[W_out-1,0],[W_out-1,H_out-1],[0,H_out-1]], dtype=np.float32)
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        rgb = cv2.merge([b, g, r])
        rgb_warp = cv2.warpPerspective(rgb, M, (W_out, H_out), flags=cv2.INTER_LINEAR, borderValue=0)
        a_warp   = cv2.warpPerspective(table_mask, M, (W_out, H_out), flags=cv2.INTER_NEAREST, borderValue=0)
        out_rgba = cv2.merge([rgb_warp[:,:,0], rgb_warp[:,:,1], rgb_warp[:,:,2], a_warp])

        # Tight crop again using alpha bbox
        ys, xs = np.where(a_warp > 0)
        if len(xs) and len(ys):
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            out_rgba = out_rgba[ymin:ymax+1, xmin:xmax+1]

        rectified_path = f"{root}.png"
        cv2.imwrite(rectified_path, out_rgba)
        print(f"Saved rectified table (keeps balls) to {rectified_path}")
    else:
        cv2.imwrite(output_path, cropped)
        print(f"Saved table-isolated image (keeps balls) to {output_path}")

# --- Main execution ---
if __name__ == "__main__":
    # List of source directories to process
    source_directories = [
        "data/photos-jpg",
    ]
    
    # Output directory for cropped images
    cropped_dir = "data/photos"
    
    # Create cropped directory if it doesn't exist
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Get all image files from multiple directories
    def get_all_image_files(directories):
        """Get all image files from multiple directories."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        all_files = []
        
        for directory in directories:
            if os.path.exists(directory):
                print(f"Scanning directory: {directory}")
                dir_files = []
                for extension in image_extensions:
                    files = glob.glob(os.path.join(directory, extension))
                    dir_files.extend(files)
                all_files.extend(dir_files)
                print(f"Found {len(dir_files)} files in {directory}")
            else:
                print(f"Directory does not exist: {directory}")
        
        return sorted(all_files)
    
    # Get all image files
    image_files = get_all_image_files(source_directories)
    
    if not image_files:
        print("No image files found in any of the specified directories.")
    else:
        print(f"Found {len(image_files)} total images to process...")
        
        for image_path in image_files:
            # Get the filename without extension
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # Create output filename with _cropped suffix
            output_filename = f"{name}.png"
            output_path = os.path.join(cropped_dir, output_filename)
            
            print(f"Processing: {filename} -> {output_filename}")
            isolate_table_keep_balls(image_path, output_path, rectify=True)
        
        print(f"Successfully processed images from multiple directories to {cropped_dir}")
