import os

# Global logging configuration
VERBOSE = True
log = print if VERBOSE else lambda *args, **kwargs: None

# Project paths configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ORIG_DATA_DIR = os.path.join(DATA_DIR, "thyroid-nodule-dataset-v1.0")
XML_DIR = os.path.join(ORIG_DATA_DIR, "xml")
JPG_DIR = os.path.join(ORIG_DATA_DIR, "jpg")

# Output directories
ENHANCE_IMGS_DIR = os.path.join(DATA_DIR, "enhance_imgs")
ANNOTATED_IMGS_DIR = os.path.join(DATA_DIR, "annotated_imgs")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
CROPPED_IMGS_DIR = os.path.join(DATA_DIR, "cropped_imgs")
DATASET_UNET_DIR = os.path.join(PROJECT_ROOT, "dataset_unet")

# U-Net Dataset directories
TRAIN_DIR = os.path.join(DATASET_UNET_DIR, "train")
TEST_DIR = os.path.join(DATASET_UNET_DIR, "test")
VAL_DIR = os.path.join(DATASET_UNET_DIR, "validation")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")