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
        dice = (2. * intersection + self.smooth) / (torch.sum(preds) + torch.sum(targets) + self.smooth)
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
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
        
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
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(preds.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(preds.device)
        
        # Get gradients
        targets_padded = F.pad(targets, (1, 1, 1, 1), mode='reflect')
        target_grad_x = F.conv2d(targets_padded, sobel_x)
        target_grad_y = F.conv2d(targets_padded, sobel_y)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Get boundary weights
        boundary_weights = torch.exp(target_grad / self.theta)
        
        # Apply weights to BCE loss
        preds = torch.sigmoid(preds)
        weighted_bce = boundary_weights * F.binary_cross_entropy(preds, targets, reduction='none')
        
        return weighted_bce.mean()

# Adaptive Loss Function that combines multiple losses
class AdaptiveLoss(nn.Module):
    def __init__(self, alpha_start=1.0, beta_start=0.0, gamma_start=0.0, decay_factor=0.99):
        super(AdaptiveLoss, self).__init__()
        self.alpha = alpha_start  # Initial weight for Dice Loss
        self.beta = beta_start    # Initial weight for BCE Loss
        self.gamma = gamma_start  # Initial weight for Focal Loss
        self.decay_factor = decay_factor
        
        # Loss components
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, preds, targets, epoch):
        # Update weights dynamically based on epoch
        self.alpha = max(0.1, self.alpha * (self.decay_factor ** epoch))  # Decay Dice weight
        self.beta = min(0.5, 1.0 - self.alpha - self.gamma)              # Increase BCE weight
        self.gamma = min(0.4, self.gamma + 0.01 * epoch)                 # Slowly increase Focal weight
        
        # Compute losses
        dice = self.dice_loss(preds, targets)
        bce = self.bce_loss(preds, targets)
        focal = self.focal_loss(preds, targets)
        boundary = self.boundary_loss(preds, targets)
        
        # Weighted sum of losses
        # As training progresses, we put more emphasis on boundary loss and focal loss
        boundary_weight = min(0.3, 0.02 * epoch)  # Gradually increase boundary loss weight
        
        loss = (self.alpha * dice + 
                self.beta * bce + 
                self.gamma * focal + 
                boundary_weight * boundary)
        
        return loss, {
            "dice_loss": dice.item(),
            "bce_loss": bce.item(),
            "focal_loss": focal.item(),
            "boundary_loss": boundary.item(),
            "alpha": self.alpha,
            "beta": self.beta, 
            "gamma": self.gamma,
            "boundary_weight": boundary_weight
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
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float().to(device)  # Ground truth (0 or 1)
    
    # Initialize adaptive loss function
    loss_fn = AdaptiveLoss().to(device)
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