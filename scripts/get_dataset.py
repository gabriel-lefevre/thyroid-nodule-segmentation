"""
Thyroid Dataset Downloader

Downloads and extracts the thyroid nodule ultrasound dataset
from GitHub releases.

Author: Gabriel Lefevre
"""

import requests
import zipfile
import os

from config import *


def download_dataset():
    """
    Download and extract thyroid ultrasound dataset.

    Downloads dataset zip from GitHub releases, extracts to specified
    directory, and cleans up temporary files.
    """
    dataset_url = "https://github.com/gabriel-lefevre/thyroid-nodule-segmentation/releases/download/dataset-v1.0/thyroid-nodule-dataset-v1.0.zip"
    dataset_path = os.path.join(DATA_DIR, "dataset.zip")

    response = requests.get(dataset_url)

    with open(dataset_path, 'wb') as f:
        f.write(response.content)

    log("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Clean up zip file
    os.remove(dataset_path)
    log("Dataset ready!")