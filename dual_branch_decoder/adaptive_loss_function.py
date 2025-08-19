import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Dice Loss Function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        intersection = torch.sum(preds * targets)
        dice = (2. * intersection + self.smooth) / (torch.sum(preds) + torch.sum(targets) + self.smooth)
        return 1 - dice  # Dice loss = 1 - Dice score

# ✅ Adaptive Loss Function
class AdaptiveLoss(nn.Module):
    def __init__(self, alpha_start=1.0, beta_start=0.0, decay_factor=0.99):
        super(AdaptiveLoss, self).__init__()
        self.alpha = alpha_start  # Initial weight for Dice Loss
        self.beta = beta_start  # Initial weight for BCE Loss
        self.decay_factor = decay_factor
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets, epoch):
        # Update weights dynamically
        self.alpha = max(0.1, self.alpha * self.decay_factor)  # Decay Dice weight
        self.beta = min(0.9, 1.0 - self.alpha)  # Increase BCE weight

        # Compute losses
        dice = self.dice_loss(preds, targets)
        bce = self.bce_loss(preds, targets)

        # Weighted sum
        loss = self.alpha * dice + self.beta * bce
        return loss

# ✅ Testing the Adaptive Loss
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample predictions & targets (Batch size = 4, 1 channel, 64x64)
    preds = torch.randn(4, 1, 64, 64).to(device)  # Model output logits
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float().to(device)  # Ground truth (0 or 1)

    # Initialize adaptive loss function
    loss_fn = AdaptiveLoss().to(device)

    # Compute loss for epoch 1
    epoch = 1
    loss = loss_fn(preds, targets, epoch)
    print(f"Epoch {epoch} - Adaptive Loss:", loss.item())
