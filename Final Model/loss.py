import torch
import torch.nn as nn
import torch.nn.functional as F


# Dice Loss Function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        intersection = torch.sum(preds * targets)
        dice = (2.0 * intersection + self.smooth) / (
            torch.sum(preds) + torch.sum(targets) + self.smooth
        )
        return 1 - dice  # Dice loss = 1 - Dice score


# Focal Loss for imbalanced classes
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        # Sigmoid first
        preds = torch.sigmoid(preds)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(preds, targets, reduction="none")

        # Calculate focal loss
        pt = targets * preds + (1 - targets) * (1 - preds)
        focal_weight = (1 - pt) ** self.gamma

        # Apply weights
        focal_loss = self.alpha * focal_weight * bce_loss

        # Return mean loss
        return focal_loss.mean()


# Boundary Loss to focus on edge regions
class BoundaryLoss(nn.Module):
    def __init__(self, theta=0.5):
        super(BoundaryLoss, self).__init__()
        self.theta = theta

    def forward(self, preds, targets):
        # Apply Sobel filter to get edges
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(preds.device)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(preds.device)
        )

        # Get gradients
        targets_padded = F.pad(targets, (1, 1, 1, 1), mode="reflect")
        target_grad_x = F.conv2d(targets_padded, sobel_x)
        target_grad_y = F.conv2d(targets_padded, sobel_y)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)

        # Get boundary weights
        boundary_weights = torch.exp(target_grad / self.theta)

        # Apply weights to BCE loss
        preds = torch.sigmoid(preds)
        weighted_bce = boundary_weights * F.binary_cross_entropy(
            preds, targets, reduction="none"
        )

        return weighted_bce.mean()


# Improved loss function for imbalanced flood detection
class ImprovedFloodLoss(nn.Module):
    def __init__(self, dice_weight=0.6, focal_weight=0.4, alpha=0.75, gamma=2.0):
        super(ImprovedFloodLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, preds, targets, epoch=None):
        dice = self.dice_loss(preds, targets)
        focal = self.focal_loss(preds, targets)

        # Combine losses
        loss = self.dice_weight * dice + self.focal_weight * focal

        return loss, {
            "dice_loss": dice.item(),
            "focal_loss": focal.item(),
            "total_loss": loss.item(),
        }


# IoU metric for evaluation
class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, preds, targets):
        # Apply sigmoid and threshold
        preds = (torch.sigmoid(preds) > self.threshold).float()

        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        # Calculate IoU
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou


# For testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample predictions & targets (Batch size = 4, 1 channel, 64x64)
    preds = torch.randn(4, 1, 64, 64).to(device)  # Model output logits
    targets = (
        torch.randint(0, 2, (4, 1, 64, 64)).float().to(device)
    )  # Ground truth (0 or 1)

    # Initialize improved flood loss function
    loss_fn = ImprovedFloodLoss().to(device)
    iou_metric = IoU().to(device)

    # Compute loss for different epochs
    for epoch in range(1, 11, 5):
        loss, loss_dict = loss_fn(preds, targets, epoch)
        iou = iou_metric(preds, targets)

        print(f"Epoch {epoch}:")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  IoU: {iou.item():.4f}")
        print(f"  Loss Components: {loss_dict}")
        print()
