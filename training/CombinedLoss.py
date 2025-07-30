import torch.nn as nn
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp



# Définition de la fonction de perte
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        dice_loss = self.dice(y_pred, y_true)
        bce_loss = self.bce(y_pred, y_true)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss

# loss_fn = CombinedLoss(alpha=0.5)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
