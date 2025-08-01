import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Combined loss function for medical image segmentation.

    Combines Dice Loss (overlap-based) with Binary Cross Entropy Loss (pixel-wise)
    for robust training on imbalanced medical datasets where background >> foreground.
    """

    def __init__(self, alpha=0.5):
        """
        Initialize combined loss with configurable weighting.

        Args:
            alpha (float): Weight for Dice loss vs BCE loss
                          alpha=0.5 means 50% Dice + 50% BCE
        """
        super(CombinedLoss, self).__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")  # Overlap-based loss
        self.bce = nn.BCEWithLogitsLoss()  # Pixel-wise classification loss
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        Calculate weighted combination of Dice and BCE losses.

        Args:
            y_pred: Model predictions (logits, before sigmoid)
            y_true: Ground truth masks (binary float values)

        Returns:
            Combined loss value for backpropagation
        """
        dice_loss = self.dice(y_pred, y_true)
        bce_loss = self.bce(y_pred, y_true)

        # Weighted combination: alpha * Dice + (1-alpha) * BCE
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss