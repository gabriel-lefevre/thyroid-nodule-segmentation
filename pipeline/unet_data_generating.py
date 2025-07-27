"""
U-Net Dataset Generation Pipeline

This module processes enhanced ultrasound images and XML annotations to generate
training-ready datasets for U-Net segmentation models.

Author: Gabriel Lefevre
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json

from config import *


class UNetDatasetGenerator:
    """
    Generate U-Net training dataset from enhanced images and XML annotations.

    Processes enhanced ultrasound images to create:
    - Cropped image patches centered on nodules
    - Corresponding binary segmentation masks
    - Annotated visualization images
    """

    def __init__(self):
        # Processing state variables
        self.img = None  # Current enhanced image (RGB)
        self.img_name = None  # Current image filename
        self.filename = None  # Output image filename
        self.points = None  # Nodule contour points (OpenCV format)
        self.center = (0, 0)  # Nodule center coordinates
        self.crop_size = 256  # Output crop dimensions

    def generate_unet_dataset(self):
        """
        Generate complete U-Net dataset from enhanced images and annotations.

        Processes all XML annotation files and creates training patches:
        1. Loads enhanced RGB images
        2. Generates binary masks from contours
        3. Creates cropped image-mask pairs
        4. Saves annotated visualizations
        """
        xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
        xml_files.sort()

        log(f"Processing {len(xml_files)} XML annotation files...")

        total_nodules = 0

        for xml_file in xml_files:
            xml_path = os.path.join(XML_DIR, xml_file)
            base_name = xml_file.replace(".xml", "")

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Process each annotated nodule
                for mark in root.findall(".//mark"):
                    image_id = mark.find("image").text
                    svg_element = mark.find("svg")

                    if svg_element is None or not svg_element.text:
                        continue

                    svg_data = svg_element.text.strip()
                    if not svg_data:
                        continue

                    self.img_name = f"{base_name}_{image_id}.jpg"
                    # Load from enhanced images directory
                    enhanced_img_path = os.path.join(ENHANCE_IMGS_DIR, self.img_name)

                    if not os.path.exists(enhanced_img_path):
                        log(f"Enhanced image {self.img_name} not found, skipping...")
                        continue

                    try:
                        points_data = json.loads(svg_data)

                        # Load enhanced RGB image
                        self.img = cv2.imread(enhanced_img_path)
                        if self.img is None:
                            log(f"Could not load enhanced image {self.img_name}")
                            continue

                        # Process each nodule in the image
                        for i, annotation in enumerate(points_data):
                            if "points" not in annotation:
                                continue

                            coordinates = annotation["points"]
                            self.points = np.array([(point['x'], point['y']) for point in coordinates], np.int32)
                            self.points = self.points.reshape((-1, 1, 2))

                            # Calculate nodule center for cropping
                            self.center = (int(np.mean(self.points[:, 0, 0])),
                                           int(np.mean(self.points[:, 0, 1])))

                            # Generate unique filename for multiple nodules per image
                            self.filename = self.img_name.replace('.jpg', f'_{i + 1}.jpg') if len(points_data) > 1 else self.img_name


                            # Generate dataset components
                            self.annotate_img()
                            self.generate_mask()
                            self.save_cropped_image()

                            total_nodules += 1

                    except json.JSONDecodeError as e:
                        log(f"Invalid JSON in {xml_file} for image {image_id}: {e}")
                        continue

            except ET.ParseError as e:
                log(f"Could not parse XML file {xml_file}: {e}")
                continue
            except Exception as e:
                log(f"Error processing {xml_file}: {e}")
                continue

        log(f"Generated {total_nodules} training samples")

    def annotate_img(self):
        """
        Create annotated visualization with nodule contours.

        Draws nodule boundaries on enhanced images for visual verification.
        Helps validate annotation quality and dataset correctness.
        """
        annotated_img = self.img.copy()

        # Draw green contour on enhanced RGB image
        cv2.polylines(annotated_img, [self.points], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        annotated_path = os.path.join(ANNOTATED_IMGS_DIR, self.filename)
        cv2.imwrite(annotated_path, annotated_img)

    def generate_mask(self):
        """
        Generate binary segmentation mask from nodule contour.

        Creates training ground truth:
        - Nodule region: white (255)
        - Background: black (0)

        Mask is cropped to match image patch dimensions.
        """
        # Create binary mask from enhanced image dimensions
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

        # Fill nodule region with white
        cv2.fillPoly(mask, [self.points], 255)

        # Crop and save mask
        cropped_mask = self.crop_img(mask)
        mask_path = os.path.join(MASKS_DIR, self.filename)
        cv2.imwrite(mask_path, cropped_mask)

    def save_cropped_image(self):
        """
        Save cropped enhanced image patch for training.

        Crops enhanced RGB image around nodule center with padding.
        Output: 256x256 RGB patch ready for U-Net input.
        """
        cropped_img = self.crop_img(self.img)
        cropped_path = os.path.join(CROPPED_IMGS_DIR, self.filename)
        cv2.imwrite(cropped_path, cropped_img)

    def crop_img(self, img):
        """
        Crop image around nodule center with intelligent padding.

        Ensures consistent 256x256 output regardless of nodule position.
        Adds black padding when crop extends beyond image boundaries.

        Args:
            img: Input image (RGB or grayscale)

        Returns:
            Cropped image with exact dimensions (256x256)
        """
        half_size = self.crop_size // 2

        # Calculate crop boundaries
        x_min, y_min = self.center[0] - half_size, self.center[1] - half_size
        x_max, y_max = self.center[0] + half_size, self.center[1] + half_size

        # Calculate required padding
        pad_x_min = max(0, -x_min)
        pad_y_min = max(0, -y_min)
        pad_x_max = max(0, x_max - img.shape[1])
        pad_y_max = max(0, y_max - img.shape[0])

        # Adjust to image boundaries
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

        # Crop and pad
        cropped = img[y_min:y_max, x_min:x_max]

        return cv2.copyMakeBorder(cropped, pad_y_min, pad_y_max, pad_x_min, pad_x_max,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
