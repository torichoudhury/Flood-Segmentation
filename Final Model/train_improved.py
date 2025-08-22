import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from model import FloodSegmentationModel
from loss import ImprovedFloodLoss, IoU
from dataset import FloodDataset
from tqdm import tqdm
import numpy as np


def main():
    # Argument parser for configurable training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    args = parser.parse_args()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss Function & Optimizer
    criterion = ImprovedFloodLoss().to(device)
    iou_metric = IoU().to(device)

    # Use a more conservative learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)

    # Learning Rate Scheduler - more aggressive
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed Precision Training
    scaler = None
    try:
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")
    except:
        print("Mixed precision not available, using standard training")

    # Checkpoint Handling
    checkpoint_path = "checkpoint_improved.pth"
    best_model_path = "best_model_improved.pth"
    start_epoch = 0
    best_loss = float("inf")

    if args.resume and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

    print(f"Starting training from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, running_iou = 0.0, 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False
        )

        for step, (sar_images, optical_images, masks) in enumerate(progress_bar):
            sar_images, optical_images, masks = (
                sar_images.to(device, non_blocking=True),
                optical_images.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(sar_images, optical_images)

                    # Resize outputs to match mask size if needed
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                # Calculate loss outside autocast
                loss, loss_dict = criterion(outputs, masks)
                iou = iou_metric(outputs, masks)

                # Backpropagation with AMP
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                outputs = model(sar_images, optical_images)

                # Resize outputs to match mask size if needed
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss, loss_dict = criterion(outputs, masks)
                iou = iou_metric(outputs, masks)

                # Standard backpropagation
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # Update learning rate
            scheduler.step()

            # Logging
            running_loss += loss.item()
            running_iou += iou.item()

            # Update progress bar
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "iou": f"{iou.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

        avg_loss = running_loss / len(train_loader)
        avg_iou = running_iou / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "avg_loss": avg_loss,
            "avg_iou": avg_iou,
        }
        torch.save(checkpoint, checkpoint_path)

        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🔹 Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")

        # Early stopping if loss becomes too small or IoU stops improving
        if avg_loss < 0.01:
            print("Loss became very small, checking for potential overfitting...")

    print("Training Complete!")
    print(f"Best loss achieved: {best_loss:.4f}")


if __name__ == "__main__":
    main()
