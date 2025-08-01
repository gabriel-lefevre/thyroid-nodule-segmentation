import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics
import segmentation_models_pytorch as smp

from config import *

from training.NoduleDataset import NoduleDataset
from training.CombinedLoss import CombinedLoss


class ModelUNetGenerator:
    """
    U-Net model generator for thyroid nodule segmentation.

    Handles model creation, training, and evaluation using segmentation_models_pytorch
    with pre-trained encoders (ResNet-50, VGG16) for medical image segmentation.
    """

    def __init__(self, model_name="resnet50"):
        """
        Initialize model, datasets, and training components.

        Args:
            model_name (str): Encoder backbone ("resnet50" or "vgg16")
        """
        # Device setup with GPU detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda':
            log(f"   GPU detected : {torch.cuda.get_device_name(0)}")
            log(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            log(f"   CUDA version : {torch.version.cuda}")
        else:
            log("Using CPU...")

        # Dataset loading with augmentation for training
        self.train_dataset = NoduleDataset(TRAIN_DIR, augment=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=0)

        self.test_dataset = NoduleDataset(TEST_DIR)
        self.test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=0)

        self.val_dataset = NoduleDataset(VAL_DIR)
        self.val_loader = DataLoader(self.val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Model creation with pre-trained encoder
        self.model_name = model_name
        self.model = smp.Unet(
            encoder_name=model_name,  # Pre-trained backbone
            encoder_weights="imagenet",  # ImageNet weights for transfer learning
            in_channels=3,  # RGB input
            classes=1,  # Binary segmentation
            activation="sigmoid"  # Output activation for probabilities
        )
        self.model.to(self.device)

        # Training components
        self.loss_fn = CombinedLoss(alpha=0.5)  # Dice + BCE loss combination
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train_model(self, num_epochs=20):
        """
        Train the U-Net model with validation monitoring.

        Args:
            num_epochs (int): Number of training epochs
        """
        best_val_loss = float("inf")
        best_model_path = os.path.join(MODELS_DIR, f"best_{self.model_name}.pth")

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                # Forward pass and optimization
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks in self.val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                    val_loss += loss.item()

            # Calculate average losses
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)

            log(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                log(f"New best model saved: {best_model_path} (Val Loss: {val_loss:.4f})")

    def evaluate_model(self):
        """
        Evaluate trained model on test set using IoU and Dice metrics.
        """
        # Initialize metrics for binary segmentation
        iou_metric = torchmetrics.JaccardIndex(task="binary", num_classes=2).to(self.device)
        dice_metric = torchmetrics.F1Score(task="binary").to(self.device)  # F1 = Dice

        self.model.eval()

        iou_score = 0
        dice_score = 0
        num_batches = 0

        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                preds = (outputs > 0.5).float()  # Binary thresholding

                # Calculate metrics for current batch
                iou = iou_metric(preds, masks.int())
                dice = dice_metric(preds, masks.int())

                iou_score += iou.item()
                dice_score += dice.item()
                num_batches += 1

        # Log final test results
        log(f"Test IoU: {iou_score / num_batches:.4f}")
        log(f"Test Dice Score: {dice_score / num_batches:.4f}")