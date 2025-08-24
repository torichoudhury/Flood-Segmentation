import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, random_split
from advanced_model import AdvancedFloodSegmentationModel, LightweightAdvancedFloodModel
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime


class AdvancedThresholdOptimizedLoss(nn.Module):
    """Enhanced loss function optimized for advanced model architecture"""

    def __init__(
        self,
        target_threshold=0.4,
        dice_weight=0.25,
        focal_weight=0.35,
        boundary_weight=0.25,
        consistency_weight=0.15,
    ):
        super(AdvancedThresholdOptimizedLoss, self).__init__()
        self.target_threshold = target_threshold
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def focal_loss(self, pred, target, alpha=0.75, gamma=2.0):
        # Use BCE with logits to avoid autocast issues
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_sigmoid = torch.sigmoid(pred)
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def boundary_aware_loss(self, pred, target):
        """Enhanced boundary loss with multi-scale edge detection"""
        pred_sigmoid = torch.sigmoid(pred)

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

        # Multi-scale boundary detection
        target_padded = F.pad(target, (1, 1, 1, 1), mode="reflect")
        target_grad_x = F.conv2d(target_padded, sobel_x)
        target_grad_y = F.conv2d(target_padded, sobel_y)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2)

        # Prediction boundaries
        pred_padded = F.pad(pred_sigmoid, (1, 1, 1, 1), mode="reflect")
        pred_grad_x = F.conv2d(pred_padded, sobel_x)
        pred_grad_y = F.conv2d(pred_padded, sobel_y)
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)

        # Boundary consistency loss
        boundary_diff = torch.abs(target_edges - pred_edges)

        # Weight boundaries more heavily
        boundary_mask = (target_edges > 0.1).float() * 3.0 + 1.0
        weighted_boundary_loss = boundary_mask * boundary_diff

        return weighted_boundary_loss.mean()

    def consistency_loss(self, pred, target):
        """Ensure prediction consistency across different scales"""
        pred_sigmoid = torch.sigmoid(pred)

        # Multi-scale consistency
        pred_small = F.interpolate(
            pred_sigmoid, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        pred_large = F.interpolate(
            pred_small,
            size=pred_sigmoid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        consistency_loss = F.mse_loss(pred_sigmoid, pred_large)

        return consistency_loss

    def threshold_optimized_loss(self, pred, target):
        """Enhanced threshold optimization for the target threshold"""
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > self.target_threshold).float()

        # Calculate confusion matrix elements
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        fn = ((1 - pred_binary) * target).sum()
        tn = ((1 - pred_binary) * (1 - target)).sum()

        # Enhanced metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        # Balanced F1 score with specificity consideration
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        balanced_accuracy = (recall + specificity) / 2

        # Combined threshold loss
        threshold_loss = 1 - (0.7 * f1 + 0.3 * balanced_accuracy)

        return threshold_loss

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_aware_loss(pred, target)
        consistency = self.consistency_loss(pred, target)
        threshold_opt = self.threshold_optimized_loss(pred, target)

        total_loss = (
            self.dice_weight * dice
            + self.focal_weight * focal
            + self.boundary_weight * boundary
            + self.consistency_weight * consistency
            + 0.2 * threshold_opt
        )

        return total_loss, {
            "dice_loss": dice.item(),
            "focal_loss": focal.item(),
            "boundary_loss": boundary.item(),
            "consistency_loss": consistency.item(),
            "threshold_loss": threshold_opt.item(),
            "total_loss": total_loss.item(),
        }


def calculate_comprehensive_metrics(pred, target, threshold=0.4):
    """Calculate comprehensive metrics with additional advanced metrics"""
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
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    # Advanced metrics
    balanced_accuracy = (recall + specificity) / 2
    mcc = ((tp * tn) - (fp * fn)) / (
        ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1e-8
    )

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "dice": dice,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def validate_model(model, val_loader, criterion, device, threshold=0.4):
    """Enhanced validation with comprehensive metrics"""
    model.eval()
    total_loss = 0.0
    all_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
        "dice": 0.0,
        "balanced_accuracy": 0.0,
        "mcc": 0.0,
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
    parser = argparse.ArgumentParser(description="Advanced Flood Segmentation Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument(
        "--model_type",
        choices=["advanced", "lightweight"],
        default="advanced",
        help="Model type to use",
    )
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
        help="Validation split ratio",
    )
    parser.add_argument(
        "--use_aspp",
        action="store_true",
        default=True,
        help="Use ASPP in advanced model",
    )
    parser.add_argument(
        "--use_dual_decoder",
        action="store_true",
        default=True,
        help="Use dual decoder in advanced model",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    # Dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")

    from dataset import FloodDataset

    full_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    dataset_size = len(full_dataset)

    # Dataset splitting
    if args.use_full_dataset:
        print(f"Using full dataset ({dataset_size} samples) for training")
        train_dataset = full_dataset
        val_size = min(20, dataset_size // 5)
        val_indices = torch.randperm(dataset_size)[:val_size]
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    else:
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

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation monitoring set: {len(val_dataset)} samples")

    # Model initialization
    if args.model_type == "advanced":
        model = AdvancedFloodSegmentationModel(
            use_aspp=args.use_aspp, use_dual_decoder=args.use_dual_decoder
        ).to(device)
        model_name = "advanced"
    else:
        model = LightweightAdvancedFloodModel().to(device)
        model_name = "lightweight"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Loss function and optimizer
    criterion = AdvancedThresholdOptimizedLoss(target_threshold=0.4).to(device)

    # Use different learning rates for different parts
    if args.model_type == "advanced":
        # Lower learning rate for pretrained backbone
        backbone_params = list(model.feature_extractor.backbone.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}
        other_params = [
            p for p in model.parameters() if id(p) not in backbone_param_ids
        ]

        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": args.lr * 0.1,
                },  # 10x lower for backbone
                {"params": other_params, "lr": args.lr},
            ],
            weight_decay=1e-4,
            eps=1e-8,
        )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
            eps=1e-8,
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    # File paths
    checkpoint_path = f"checkpoint_{model_name}_advanced.pth"
    best_model_path = f"best_model_{model_name}_advanced.pth"
    history_file = f"training_history_{model_name}_advanced.json"

    # Training state
    start_epoch = 0
    best_f1 = 0.0
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
        "train_specificity": [],
        "val_specificity": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_dice": [],
        "val_dice": [],
        "train_balanced_accuracy": [],
        "val_balanced_accuracy": [],
        "train_mcc": [],
        "val_mcc": [],
        "learning_rate": [],
    }

    # Resume training if requested
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        if "training_history" in checkpoint:
            training_history = checkpoint["training_history"]

    print(f"Starting advanced training from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "accuracy": 0.0,
            "dice": 0.0,
            "balanced_accuracy": 0.0,
            "mcc": 0.0,
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
            with torch.amp.autocast("cuda"):
                outputs = model(sar_images, optical_images)

                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Loss calculation
                loss, loss_dict = criterion(outputs, masks)

            # Metrics calculation
            metrics = calculate_comprehensive_metrics(outputs, masks, threshold=0.4)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    "mcc": f"{metrics['mcc']:.4f}",
                }
            )

        # Average training metrics
        avg_loss = running_loss / len(train_loader)
        for key in all_metrics:
            all_metrics[key] /= len(train_loader)

        # Validation phase
        print(f"Running validation...")
        val_loss, val_metrics = validate_model(model, val_loader, criterion, device)

        # Update learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Store history
        training_history["train_loss"].append(avg_loss)
        training_history["val_loss"].append(val_loss)
        training_history["learning_rate"].append(current_lr)

        for key in all_metrics:
            training_history[f"train_{key}"].append(all_metrics[key])
            training_history[f"val_{key}"].append(val_metrics[key])

        # Epoch summary with advanced metrics
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train F1: {all_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(
            f"  Train IoU: {all_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}"
        )
        print(
            f"  Train MCC: {all_metrics['mcc']:.4f} | Val MCC: {val_metrics['mcc']:.4f}"
        )
        print(
            f"  Train Bal.Acc: {all_metrics['balanced_accuracy']:.4f} | Val Bal.Acc: {val_metrics['balanced_accuracy']:.4f}"
        )
        print(f"  LR: {current_lr:.2e}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "training_history": training_history,
            "train_metrics": all_metrics,
            "val_metrics": val_metrics,
            "model_config": {
                "model_type": args.model_type,
                "use_aspp": args.use_aspp if args.model_type == "advanced" else None,
                "use_dual_decoder": (
                    args.use_dual_decoder if args.model_type == "advanced" else None
                ),
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if args.use_full_dataset:
            current_f1 = all_metrics["f1"]
        else:
            current_f1 = val_metrics["f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 New best F1: {best_f1:.4f}")

        # Save training history
        with open(history_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "epochs_completed": epoch + 1,
                    "best_f1": best_f1,
                    "model_config": checkpoint["model_config"],
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
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

    print(f"\nAdvanced training complete!")
    print(f"Model: {model_name}")
    print(f"Best F1 achieved: {best_f1:.4f}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Training history saved to: {history_file}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
