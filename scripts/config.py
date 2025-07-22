import os

# Project paths configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "thyroid-nodule-dataset-v1.0")
XML_DIR = os.path.join(DATA_DIR, "xml")
JPG_DIR = os.path.join(DATA_DIR, "jpg")

# Output directories
ANNOTATED_IMGS_DIR = os.path.join(PROJECT_ROOT, "data", "annotated_imgs")
PROCESSED_IMGS_DIR = os.path.join(PROJECT_ROOT, "data", "processed_imgs")
MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "masks")
CROPPED_MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "cropped_masks")

# Annotation settings
CONTOUR_COLOR = (0, 0, 255)  # Red in BGR
CONTOUR_THICKNESS = 2