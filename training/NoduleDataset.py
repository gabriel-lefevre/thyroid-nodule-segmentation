import os
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np


class NoduleDataset(Dataset):
    """
    Custom dataset for thyroid nodule segmentation.

    Loads enhanced ultrasound images and corresponding binary masks,
    applies data augmentation for training, and converts to PyTorch tensors.
    """

    def __init__(self, data_dir, img_size=(256, 256), augment=False):
        """
        Initialize dataset from directory structure.

        Args:
            data_dir (str): Path to dataset split (train/validation/test)
            img_size (tuple): Target image dimensions
            augment (bool): Apply data augmentation for training
        """
        # Directory structure: data_dir/{images,masks}/
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        # Load and sort filenames for consistent pairing
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        self.img_size = img_size
        self.augment = augment

        # Data augmentation pipeline for training robustness
        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),  # Mirror augmentation
            A.RandomBrightnessContrast(p=0.2),  # Intensity variations
            A.Rotate(limit=15, p=0.5)  # Small rotations
        ]) if augment else None

    def __len__(self):
        """Return total number of image-mask pairs."""
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        Load and preprocess image-mask pair at given index.

        Args:
            index (int): Sample index

        Returns:
            tuple: (image_tensor, mask_tensor) ready for model input
                  - image: (3, 256, 256) RGB tensor, normalized [0,1]
                  - mask: (1, 256, 256) binary tensor, normalized [0,1]
        """
        img_name = self.image_filenames[index]
        mask_name = self.mask_filenames[index]

        # Load and resize image to target dimensions
        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img = cv2.resize(img, self.img_size) / 255.0  # Normalize to [0,1]

        # Load mask as grayscale and normalize
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size) / 255.0  # Normalize to [0,1]

        # Apply augmentations if enabled (training only)
        if self.augmentations:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # Convert to PyTorch format: HWC -> CHW for images
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension: (H,W) -> (1,H,W)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)