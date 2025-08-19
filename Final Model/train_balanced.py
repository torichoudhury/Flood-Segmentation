import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from model import FloodSegmentationModel
from dataset import FloodDataset
from tqdm import tqdm


# Balanced Loss Function
class BalancedLoss(nn.Module):
    def __init__(self):
        super(BalancedLoss, self).__init__()
        # Use lower focal loss alpha and gamma to reduce over-prediction
        self.focal_alpha = 0.25
        self.focal_gamma = 1.0  # Reduced from 2.0

        # Dice loss for overlap
        self.dice_smooth = 1.0

    def focal_loss(self, preds, targets):
        """Focal loss with reduced aggressiveness."""
        preds = torch.sigmoid(preds)
        bce_loss = F.binary_cross_entropy(preds, targets, reduction="none")
        pt = targets * preds + (1 - targets) * (1 - preds)
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = self.focal_alpha * focal_weight * bce_loss
        return focal_loss.mean()

    def dice_loss(self, preds, targets):
        """Dice loss for overlap."""
        preds = torch.sigmoid(preds)
        intersection = torch.sum(preds * targets)
        dice = (2.0 * intersection + self.dice_smooth) / (
            torch.sum(preds) + torch.sum(targets) + self.dice_smooth
        )
        return 1 - dice

    def forward(self, preds, targets, epoch=0):
        # Simple weighted combination - no complex scheduling
        dice = self.dice_loss(preds, targets)
        focal = self.focal_loss(preds, targets)

        # Balanced weights
        total_loss = 0.6 * dice + 0.4 * focal

        return total_loss, {"dice": dice.item(), "focal": focal.item()}


# IoU Metric
class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, preds, targets):
        # Apply sigmoid and threshold
        preds = torch.sigmoid(preds)
        preds_binary = (preds > self.threshold).float()

        # Calculate IoU
        intersection = torch.sum(preds_binary * targets)
        union = torch.sum(preds_binary) + torch.sum(targets) - intersection
        iou = intersection / (union + 1e-8)

        return iou


def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate (reduced)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    args = parser.parse_args()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load Dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")
    train_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Model Initialization
    model = FloodSegmentationModel().to(device)

    # Balanced Loss Function & Metrics
    criterion = BalancedLoss().to(device)
    iou_metric = IoU(threshold=0.5).to(
        device
    )  # Start with 0.5, we'll adjust based on validation

    # Optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Mixed Precision Training
    scaler = torch.amp.GradScaler("cuda")

    # Checkpoint Handling
    checkpoint_path = "balanced_checkpoint.pth"
    best_model_path = "balanced_best_model.pth"
    start_epoch = 0
    best_iou = 0.0

    if args.resume and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_iou", 0.0)

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_focal = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False
        )

        for step, (sar_images, optical_images, masks) in enumerate(progress_bar):
            sar_images, optical_images, masks = (
                sar_images.to(device),
                optical_images.to(device),
                masks.to(device),
            )

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda"):
                outputs = model(sar_images, optical_images)

                # Resize outputs to match mask size
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            # Calculate loss outside autocast
            loss, loss_dict = criterion(outputs, masks, epoch)
            iou = iou_metric(outputs, masks)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            running_loss += loss.item()
            running_iou += iou.item()
            running_dice += loss_dict["dice"]
            running_focal += loss_dict["focal"]

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "iou": f"{iou.item():.4f}",
                    "dice": f'{loss_dict["dice"]:.4f}',
                    "focal": f'{loss_dict["focal"]:.4f}',
                }
            )

        # Epoch statistics
        avg_loss = running_loss / len(train_loader)
        avg_iou = running_iou / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        avg_focal = running_focal / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] - "
            f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, "
            f"Dice: {avg_dice:.4f}, Focal: {avg_focal:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_iou": best_iou,
                "loss": avg_loss,
            },
            checkpoint_path,
        )

        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"🔹 Best model updated at epoch {epoch+1} with IoU {best_iou:.4f}")

    print("Training Complete!")
    print(f"Best IoU achieved: {best_iou:.4f}")


if __name__ == "__main__":
    main()
