from config import *

from scripts.get_dataset import download_dataset
from scripts.enhance_dataset import enhance_images
from scripts.split_dataset import split_dataset
from pipeline.unet_data_generating import UNetDatasetGenerator
from training.ModelUNetGenerator import ModelUNetGenerator


def _setup_workspace():
    """Setup workspace directories for datasets and output."""
    directories = [
        DATA_DIR,
        ENHANCE_IMGS_DIR,
        ANNOTATED_IMGS_DIR,
        MASKS_DIR,
        CROPPED_IMGS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    log("Starting thyroid nodule preprocessing pipeline...")

    log("Setting up workspace directories...")
    _setup_workspace()

    log("Downloading thyroid dataset...")
    download_dataset()

    log("Enhancing ultrasound images...")
    enhance_images()

    log("Generating Dataset U-NET...")
    dataset_generator = UNetDatasetGenerator()
    dataset_generator.generate_unet_dataset()
    success = split_dataset()

    if success:
        log("U-Net Dataset generated successfully!")
    else:
        log("U-net Dataset generation failed!")
        exit(1)

    resnet50_model = ModelUNetGenerator(model_name="resnet50")
    resnet50_model.train_model()
    resnet50_model.evaluate_model()

    vgg16_model = ModelUNetGenerator(model_name="vgg16")
    vgg16_model.train_model()
    vgg16_model.evaluate_model()


if __name__ == "__main__":
    main()