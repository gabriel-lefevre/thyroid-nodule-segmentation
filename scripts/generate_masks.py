import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json

from config import XML_DIR, JPG_DIR, ANNOTATED_IMGS_DIR, MASKS_DIR, CONTOUR_COLOR, CONTOUR_THICKNESS


def generate_mask(points, jpg_path, mask_name):
    """
    Generate binary mask from annotation points and save to masks directory.
    """
    # Load original image to get dimensions
    original_image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        return False

    # Create black mask with same dimensions
    mask = np.zeros_like(original_image, dtype=np.uint8)

    # Draw white shape
    cv2.fillPoly(mask, [points], 255)

    # Save mask as JPG
    mask_path = os.path.join(MASKS_DIR, mask_name)
    cv2.imwrite(mask_path, mask)

    return True


def annotate_img(image, points):
    """
    Draw red contour on image for annotation visualization.
    """
    cv2.polylines(image, [points], isClosed=True, color=CONTOUR_COLOR, thickness=CONTOUR_THICKNESS)


def generate_data():
    """
    Main function to generate masks and annotated images from XML annotations.
    """
    # Create output directories
    os.makedirs(ANNOTATED_IMGS_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    print("Starting mask generation and image annotation process...")

    # Get all XML files
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
    xml_files.sort()

    total_images_processed = 0
    total_nodules_processed = 0

    # Process each XML file
    for xml_file in xml_files:
        xml_path = os.path.join(XML_DIR, xml_file)
        base_name = xml_file.replace(".xml", "")

        try:
            # Parse XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find all annotations in the XML
            for mark in root.findall(".//mark"):
                image_id = mark.find("image").text
                svg_element = mark.find("svg")

                if svg_element is None or not svg_element.text:
                    continue

                svg_data = svg_element.text.strip()
                if not svg_data:
                    continue

                # Construct image filename
                jpg_name = f"{base_name}_{image_id}.jpg"
                jpg_path = os.path.join(JPG_DIR, jpg_name)

                # Check if JPG file exists
                if not os.path.exists(jpg_path):
                    print(f"Warning: Image {jpg_name} not found, skipping...")
                    continue

                try:
                    # Parse JSON coordinates from XML
                    points_data = json.loads(svg_data)

                    # Load the image
                    image = cv2.imread(jpg_path)
                    if image is None:
                        print(f"Error: Could not load image {jpg_name}")
                        continue

                    nodules_in_image = 0

                    # Process each nodule annotation in the image
                    for annotation in points_data:
                        if "points" not in annotation:
                            continue

                        # Extract coordinates and convert to OpenCV format
                        coordinates = annotation["points"]
                        points = np.array([(point['x'], point['y']) for point in coordinates], np.int32)
                        points = points.reshape((-1, 1, 2))

                        # Generate mask for this nodule
                        generate_mask(points, jpg_path, jpg_name)

                        # Annotate image with contour
                        annotate_img(image, points)

                        nodules_in_image += 1
                        total_nodules_processed += 1

                    # Save annotated image as JPG
                    annotated_path = os.path.join(ANNOTATED_IMGS_DIR, jpg_name)
                    cv2.imwrite(annotated_path, image)

                    total_images_processed += 1
                    print(f"Processed {jpg_name}: {nodules_in_image} nodule(s) processed")

                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in {xml_file} for image {image_id}: {e}")
                    continue

        except ET.ParseError as e:
            print(f"Error: Could not parse XML file {xml_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

    # Print summary
    print(f"\nData generation completed!")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total nodules processed: {total_nodules_processed}")
    print(f"Annotated images saved in: {ANNOTATED_IMGS_DIR}")
    print(f"Masks saved in: {MASKS_DIR}")


if __name__ == "__main__":
    generate_data()