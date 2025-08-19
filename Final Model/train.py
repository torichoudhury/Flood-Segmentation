import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from model import FloodSegmentationModel
from loss import AdaptiveLoss, IoU
from dataset import FloodDataset
from tqdm import tqdm


def main():
    # ✅ Argument parser for configurable training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=2, help="Gradient accumulation steps"
    )
    args = parser.parse_args()

    # ✅ Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Optimize CNN performance

    # ✅ Load Dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")
    train_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
    )

    # ✅ Model Initialization
    model = FloodSegmentationModel().to(device)

    # ✅ Loss Function & Optimizer
    criterion = AdaptiveLoss().to(device)
    iou_metric = IoU().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ✅ Mixed Precision Training (AMP)
    scaler = torch.amp.GradScaler("cuda")  # Updated to new API

    # ✅ Checkpoint Handling
    checkpoint_path = "checkpoint.pth"
    best_model_path = "best_model.pth"
    start_epoch = 0
    best_loss = float("inf")

    if args.resume and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

    # ✅ Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, running_iou = 0.0, 0.0
        optimizer.zero_grad()  # Accumulate gradients

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False
        )

        for step, (sar_images, optical_images, masks) in enumerate(progress_bar):
            sar_images, optical_images, masks = (
                sar_images.to(device),
                optical_images.to(device),
                masks.to(device),
            )

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda"):
                outputs = model(
                    sar_images, optical_images
                )  # Pass both SAR and optical images

                # Resize outputs to match mask size if needed
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            # Calculate loss outside autocast to avoid BCE issues
            loss, loss_dict = criterion(outputs, masks, epoch)
            iou = iou_metric(outputs, masks)

            # Backpropagation with AMP
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % args.accumulation_steps == 0 or step == len(
                train_loader
            ) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Logging
            running_loss += loss.item()
            running_iou += iou.item()
            progress_bar.set_postfix(loss=loss.item(), iou=iou.item())

        avg_loss = running_loss / len(train_loader)
        avg_iou = running_iou / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}"
        )

        # ✅ Save Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            checkpoint_path,
        )

        # ✅ Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🔹 Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")

    print("Training Complete!")


if __name__ == "__main__":
    main()
