import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, random_split
from model import FloodSegmentationModel
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime


class ThresholdOptimizedLoss(nn.Module):
    """Loss function optimized for the specific threshold (0.4) identified from validation"""

    def __init__(
        self,
        target_threshold=0.4,
        dice_weight=0.3,
        focal_weight=0.4,
        boundary_weight=0.3,
    ):
        super(ThresholdOptimizedLoss, self).__init__()
        self.target_threshold = target_threshold
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def focal_loss(self, pred, target, alpha=0.75, gamma=2.0):
        pred = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
        pt = target * pred + (1 - target) * (1 - pred)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def threshold_optimized_loss(self, pred, target):
        """Loss that optimizes for performance at the target threshold"""
        pred_sigmoid = torch.sigmoid(pred)

        # Create predictions at target threshold
        pred_binary = (pred_sigmoid > self.target_threshold).float()

        # Penalize errors at the target threshold more heavily
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        fn = ((1 - pred_binary) * target).sum()

        # Precision and recall at threshold
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # F1-based loss (higher F1 = lower loss)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        threshold_loss = 1 - f1

        return threshold_loss

    def boundary_aware_loss(self, pred, target):
        """Focus on boundary regions where classification is most critical"""
        # Sobel edge detection
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(pred.device)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(pred.device)
        )

        target_padded = F.pad(target, (1, 1, 1, 1), mode="reflect")
        target_grad_x = F.conv2d(target_padded, sobel_x)
        target_grad_y = F.conv2d(target_padded, sobel_y)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2)

        # Weight areas with edges more heavily
        boundary_mask = (
            target_edges > 0.1
        ).float() * 2.0 + 1.0  # 3x weight for boundaries
        pred_sigmoid = torch.sigmoid(pred)

        weighted_bce = boundary_mask * F.binary_cross_entropy(
            pred_sigmoid, target, reduction="none"
        )
        return weighted_bce.mean()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        threshold_opt = self.threshold_optimized_loss(pred, target)
        boundary = self.boundary_aware_loss(pred, target)

        total_loss = (
            self.dice_weight * dice
            + self.focal_weight * focal
            + self.boundary_weight * boundary
            + 0.3 * threshold_opt
        )  # Add threshold optimization component

        return total_loss, {
            "dice_loss": dice.item(),
            "focal_loss": focal.item(),
            "boundary_loss": boundary.item(),
            "threshold_loss": threshold_opt.item(),
            "total_loss": total_loss.item(),
        }


class AdvancedIoU(nn.Module):
    def __init__(self, threshold=0.4):
        super(AdvancedIoU, self).__init__()
        self.threshold = threshold

    def forward(self, preds, targets):
        preds = (torch.sigmoid(preds) > self.threshold).float()

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou


def calculate_comprehensive_metrics(pred, target, threshold=0.4):
    """Calculate all relevant metrics"""
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()

    # Basic confusion matrix elements
    tp = ((pred_binary == 1) & (target == 1)).sum().item()
    tn = ((pred_binary == 0) & (target == 0)).sum().item()
    fp = ((pred_binary == 1) & (target == 0)).sum().item()
    fn = ((pred_binary == 0) & (target == 1)).sum().item()

    # Calculate metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "dice": dice,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def validate_model(model, val_loader, criterion, device, threshold=0.4):
    """Validate the model and return average metrics"""
    model.eval()
    total_loss = 0.0
    all_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
        "dice": 0.0,
    }

    with torch.no_grad():
        for sar_images, optical_images, masks in val_loader:
            sar_images = sar_images.to(device, non_blocking=True)
            optical_images = optical_images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            outputs = model(sar_images, optical_images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Loss calculation
            loss, _ = criterion(outputs, masks)
            total_loss += loss.item()

            # Metrics calculation
            metrics = calculate_comprehensive_metrics(
                outputs, masks, threshold=threshold
            )
            for key in all_metrics:
                all_metrics[key] += metrics[key]

    # Average metrics
    avg_loss = total_loss / len(val_loader)
    for key in all_metrics:
        all_metrics[key] /= len(val_loader)

    return avg_loss, all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument(
        "--use_full_dataset",
        action="store_true",
        default=True,
        help="Use full dataset for training",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (only if not using full dataset)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")

    from dataset import FloodDataset

    full_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    dataset_size = len(full_dataset)

    if args.use_full_dataset:
        # Use full dataset for training, create a smaller validation set for monitoring
        print(f"Using full dataset ({dataset_size} samples) for training")
        train_dataset = full_dataset

        # Create a small validation subset (just for monitoring, not for model selection)
        val_size = min(
            20, dataset_size // 5
        )  # Use 20 samples or 20% of data, whichever is smaller
        val_indices = torch.randperm(dataset_size)[:val_size]
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation monitoring set: {len(val_dataset)} samples")
    else:
        # Split dataset into train and validation
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print(
            f"Dataset split - Train: {len(train_dataset)}, Validation: {len(val_dataset)}"
        )

    # Load the existing best model
    model = FloodSegmentationModel().to(device)
    if os.path.exists("best_model_improved.pth"):
        print("Loading best model as starting point...")
        model.load_state_dict(
            torch.load("best_model_improved.pth", map_location=device)
        )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Threshold-optimized loss
    criterion = ThresholdOptimizedLoss(target_threshold=0.4).to(device)
    iou_metric = AdvancedIoU(threshold=0.4).to(device)

    # Conservative optimizer for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-8)

    # Gentle learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Checkpoints and history tracking
    checkpoint_path = "checkpoint_optimized.pth"
    best_model_path = "best_model_optimized.pth"
    start_epoch = 0
    best_f1 = 0.0

    # Training history
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_iou": [],
        "val_iou": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_dice": [],
        "val_dice": [],
        "learning_rate": [],
    }

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        if "training_history" in checkpoint:
            training_history = checkpoint["training_history"]

    print(f"Starting optimized fine-tuning from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "accuracy": 0.0,
            "dice": 0.0,
        }

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False
        )

        for step, (sar_images, optical_images, masks) in enumerate(progress_bar):
            sar_images = sar_images.to(device, non_blocking=True)
            optical_images = optical_images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(sar_images, optical_images)

                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            # Loss and metrics calculation
            loss, loss_dict = criterion(outputs, masks)
            metrics = calculate_comprehensive_metrics(outputs, masks, threshold=0.4)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate metrics
            running_loss += loss.item()
            for key in all_metrics:
                all_metrics[key] += metrics[key]

            # Progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "f1": f"{metrics['f1']:.4f}",
                    "iou": f"{metrics['iou']:.4f}",
                }
            )

        # Average training metrics
        avg_loss = running_loss / len(train_loader)
        for key in all_metrics:
            all_metrics[key] /= len(train_loader)

        # Validation phase (monitoring only)
        print(f"Running validation monitoring...")
        val_loss, val_metrics = validate_model(model, val_loader, criterion, device)

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Store history
        training_history["train_loss"].append(avg_loss)
        training_history["val_loss"].append(val_loss)
        training_history["learning_rate"].append(current_lr)

        for key in all_metrics:
            training_history[f"train_{key}"].append(all_metrics[key])
            training_history[f"val_{key}"].append(val_metrics[key])

        # Epoch summary
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train F1: {all_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(
            f"  Train IoU: {all_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}"
        )
        print(
            f"  Train Precision: {all_metrics['precision']:.4f} | Val Precision: {val_metrics['precision']:.4f}"
        )
        print(
            f"  Train Recall: {all_metrics['recall']:.4f} | Val Recall: {val_metrics['recall']:.4f}"
        )
        print(f"  LR: {current_lr:.2e}")

        # Save checkpoint with training history
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "training_history": training_history,
            "train_metrics": all_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model based on training F1 (since we're using full dataset)
        if args.use_full_dataset:
            current_f1 = all_metrics["f1"]  # Use training F1 for full dataset
        else:
            current_f1 = val_metrics["f1"]  # Use validation F1 for split dataset

        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  🔹 New best F1: {best_f1:.4f}")

        # Save training history as JSON
        history_file = "training_history_optimized.json"
        with open(history_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "epochs_completed": epoch + 1,
                    "best_f1": best_f1,
                    "use_full_dataset": args.use_full_dataset,
                    "dataset_info": {
                        "total_samples": dataset_size,
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                    },
                    "history": training_history,
                },
                f,
                indent=2,
            )

    print(f"\nOptimized fine-tuning complete!")
    print(f"Best F1 achieved: {best_f1:.4f}")
    print(f"Training history saved to: training_history_optimized.json")
    print(f"Checkpoint with history saved to: {checkpoint_path}")
    print(f"Best model saved to: {best_model_path}")

    if args.use_full_dataset:
        print(f"✨ Trained on full dataset ({len(train_dataset)} samples)")
    else:
        print(
            f"Trained on split dataset (Train: {len(train_dataset)}, Val: {len(val_dataset)})"
        )


if __name__ == "__main__":
    main()
