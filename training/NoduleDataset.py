import os
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp


class NoduleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), augment=False):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5)
        ]) if augment else None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        mask_name = self.mask_filenames[index]


        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img = cv2.resize(img, self.img_size) / 255.0  # Normalisation
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size) / 255.0  # Normalisation

        if self.augmentations:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = np.transpose(img, (2, 0, 1))  # Convertir en format PyTorch (C, H, W)
        mask = np.expand_dims(mask, axis=0)  # Ajouter une dimension pour le masque

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
