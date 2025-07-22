import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json

from config import XML_DIR, JPG_DIR, MASKS_DIR, PROCESSED_IMGS_DIR, CROPPED_MASKS_DIR


def apply_filters(image):
    """
    Apply CLAHE and Non-Local Means filtering to enhance ultrasound image quality.
    CLAHE improves contrast, NLM reduces noise while preserving edges.
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit=2.0 prevents over-amplification, tileGridSize=(8,8) divides image into 8x8 grid
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(image)

    # Apply Non-Local Means denoising to reduce ultrasound speckle noise
    # h=10 controls denoising strength, templateWindowSize=7 and searchWindowSize=21 define neighborhood sizes
    filtered_img = cv2.fastNlMeansDenoising(clahe_img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert grayscale to RGB for compatibility with training pipeline
    return cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)


def crop_with_padding(img, center_x, center_y, crop_size=256):
    """
    Crop image around nodule center with black padding if crop extends beyond image bounds.
    Ensures output is always exactly crop_size x crop_size pixels.
    """
    half_size = crop_size // 2

    # Calculate crop boundaries around center
    x_min, y_min = center_x - half_size, center_y - half_size
    x_max, y_max = center_x + half_size, center_y + half_size

    # Calculate padding needed if crop goes outside image boundaries
    pad_x_min = max(0, -x_min)  # Left padding
    pad_y_min = max(0, -y_min)  # Top padding
    pad_x_max = max(0, x_max - img.shape[1])  # Right padding
    pad_y_max = max(0, y_max - img.shape[0])  # Bottom padding

    # Adjust crop coordinates to stay within image bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

    # Perform the crop
    cropped = img[y_min:y_max, x_min:x_max]

    # Add black padding to reach exact crop_size dimensions
    return cv2.copyMakeBorder(cropped, pad_y_min, pad_y_max, pad_x_min, pad_x_max, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def preprocess_data():
    """
    Main function to crop and filter all images and masks around nodule centers.
    Processes XML annotations to extract nodule coordinates, then crops both
    original images and masks to 256x256 patches centered on each nodule.
    """
    # Create output directories for processed data
    os.makedirs(PROCESSED_IMGS_DIR, exist_ok=True)
    os.makedirs(CROPPED_MASKS_DIR, exist_ok=True)

    print("Starting preprocessing...")

    # Get all XML annotation files
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
    total_processed = 0

    # Process each XML file to extract nodule locations
    for xml_file in xml_files:
        xml_path = os.path.join(XML_DIR, xml_file)
        base_name = xml_file.replace(".xml", "")

        # Parse XML to extract annotations
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Process each marked nodule in the XML
        for mark in root.findall(".//mark"):
            image_id = mark.find("image").text
            svg_data = mark.find("svg").text

            if not svg_data:
                continue

            # Construct paths to original image and generated mask
            jpg_name = f"{base_name}_{image_id}.jpg"
            image_path = os.path.join(JPG_DIR, jpg_name)
            mask_path = os.path.join(MASKS_DIR, jpg_name)

            # Skip if either image or mask is missing
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                continue

            # Parse JSON coordinates from XML annotation
            points_data = json.loads(svg_data)

            # Process each nodule annotation (can be multiple per image)
            for i, annotation in enumerate(points_data):
                if "points" not in annotation:
                    continue

                # Extract nodule boundary coordinates
                coordinates = annotation["points"]
                points = np.array([(point['x'], point['y']) for point in coordinates], np.int32)

                # Calculate nodule center for cropping
                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))

                # Load original ultrasound image and crop around nodule
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                cropped_image = crop_with_padding(image, center_x, center_y)

                # Apply enhancement filters to cropped image
                filtered_image = apply_filters(cropped_image)

                # Load corresponding mask and crop at same location
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                cropped_mask = crop_with_padding(mask, center_x, center_y)

                # Generate unique filename for multiple nodules per image
                output_name = f"{base_name}_{image_id}_{i + 1}.jpg" if len(points_data) > 1 else jpg_name

                # Save processed image and cropped mask
                cv2.imwrite(os.path.join(PROCESSED_IMGS_DIR, output_name), filtered_image)
                cv2.imwrite(os.path.join(CROPPED_MASKS_DIR, output_name), cropped_mask)

                total_processed += 1

    print(f"Preprocessing completed! {total_processed} nodules processed.")
    print(f"Processed images saved in: {PROCESSED_IMGS_DIR}")
    print(f"Cropped masks saved in: {CROPPED_MASKS_DIR}")


if __name__ == "__main__":
    preprocess_data()