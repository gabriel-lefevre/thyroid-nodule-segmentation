import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from training.NoduleDataset import NoduleDataset
from training.CombinedLoss import CombinedLoss


class ModelUNetGenerator:
    def __init__(self):
        self.dataset = NoduleDataset(...)
        self.loss_fn = CombinedLoss(...)