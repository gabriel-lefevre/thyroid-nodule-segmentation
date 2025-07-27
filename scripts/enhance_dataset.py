import os
import cv2

from config import *


def enhance_images():
    """
    Apply image enhancement to all ultrasound images.

    Enhancement pipeline:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
       - Improves local contrast while preventing over-amplification
       - Configured with clipLimit=2.0 and 8x8 tile grid

    2. Non-Local Means Denoising
       - Reduces ultrasound speckle noise while preserving edges
       - Uses h=10 for denoising strength, optimized window sizes

    3. Grayscale to RGB conversion for model compatibility
    """
    # Get all JPG files
    jpg_files = [f for f in os.listdir(JPG_DIR) if f.lower().endswith('.jpg')]

    for i, jpg_file in enumerate(jpg_files):
        # Load image
        img_path = os.path.join(JPG_DIR, jpg_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Could not load {jpg_file}")
            continue

        # Apply CLAHE for adaptive contrast enhancement
        # clipLimit=2.0 prevents over-amplification of noise
        # tileGridSize=(8,8) divides image into 8x8 processing tiles
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img)

        # Apply Non-Local Means denoising to reduce ultrasound speckle
        # h=10: controls denoising strength (higher = more denoising)
        # templateWindowSize=7: size of template patch for comparison
        # searchWindowSize=21: size of search area for similar patches
        filtered_img = cv2.fastNlMeansDenoising(clahe_img, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Save enhanced image
        output_path = os.path.join(ENHANCE_IMGS_DIR, jpg_file)
        cv2.imwrite(output_path, filtered_img)

        if i % 100 == 0:
            log(f"Processed {i}/{len(jpg_files)} images")

    log(f"Enhancement completed! {len(jpg_files)} images processed and saved to {ENHANCE_IMGS_DIR}.")
