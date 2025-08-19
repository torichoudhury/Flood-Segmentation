import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import FloodDataset
from model import FloodSegmentationModel
import torch.nn.functional as F
import matplotlib.pyplot as plt


def analyze_dataset_and_model():
    """Comprehensive analysis of dataset and model behavior."""

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")

    dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Load Model
    model = FloodSegmentationModel().to(device)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("✅ Loaded trained model")
    else:
        print("⚠️  Using untrained model")
    model.eval()

    print("🔍 DATASET ANALYSIS")
    print("=" * 50)

    # Analyze first 10 batches
    total_pixels = 0
    total_flood_pixels = 0
    model_outputs = []
    ground_truths = []

    with torch.no_grad():
        for batch_idx, (sar, optical, masks) in enumerate(dataloader):
            if batch_idx >= 10:  # Analyze first 10 batches
                break

            sar, optical, masks = sar.to(device), optical.to(device), masks.to(device)

            # Dataset statistics
            batch_pixels = masks.numel()
            batch_flood_pixels = masks.sum().item()

            total_pixels += batch_pixels
            total_flood_pixels += batch_flood_pixels

            print(f"Batch {batch_idx+1}:")
            print(f"  Total pixels: {batch_pixels:,}")
            print(f"  Flood pixels: {int(batch_flood_pixels):,}")
            print(f"  Flood ratio: {batch_flood_pixels/batch_pixels*100:.4f}%")
            print(f"  Mask range: [{masks.min().item():.4f}, {masks.max().item():.4f}]")

            # Model predictions
            outputs = model(sar, optical)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            predictions = torch.sigmoid(outputs)
            print(
                f"  Model output range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]"
            )

            # Store for further analysis
            model_outputs.append(predictions.cpu())
            ground_truths.append(masks.cpu())

    # Overall dataset statistics
    flood_ratio = total_flood_pixels / total_pixels
    print(f"\n📊 OVERALL DATASET STATISTICS")
    print(f"=" * 50)
    print(f"Total pixels analyzed: {total_pixels:,}")
    print(f"Total flood pixels: {int(total_flood_pixels):,}")
    print(f"Overall flood ratio: {flood_ratio*100:.4f}%")

    if flood_ratio < 0.01:
        print("⚠️  WARNING: Very low flood ratio (<1%) - severe class imbalance!")
    elif flood_ratio < 0.05:
        print("⚠️  WARNING: Low flood ratio (<5%) - class imbalance issue")

    # Model prediction analysis
    all_outputs = torch.cat(model_outputs, dim=0)
    all_masks = torch.cat(ground_truths, dim=0)

    print(f"\n🤖 MODEL PREDICTION ANALYSIS")
    print(f"=" * 50)
    print(f"Prediction range: [{all_outputs.min():.4f}, {all_outputs.max():.4f}]")
    print(f"Prediction mean: {all_outputs.mean():.4f}")
    print(f"Prediction std: {all_outputs.std():.4f}")

    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n🎯 THRESHOLD ANALYSIS")
    print(f"=" * 50)
    print("Threshold | Predicted Flood Pixels | IoU    | F1-Score")
    print("-" * 50)

    best_iou = 0
    best_threshold = 0.5

    for threshold in thresholds:
        pred_binary = (all_outputs > threshold).float()

        # Calculate metrics
        tp = ((pred_binary == 1) & (all_masks == 1)).sum().item()
        fp = ((pred_binary == 1) & (all_masks == 0)).sum().item()
        fn = ((pred_binary == 0) & (all_masks == 1)).sum().item()

        predicted_flood = pred_binary.sum().item()

        # IoU and F1
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"{threshold:9.1f} | {predicted_flood:18,} | {iou:6.4f} | {f1:8.4f}")

        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold

    print(f"\n✨ RECOMMENDATIONS")
    print(f"=" * 50)
    print(f"Best threshold: {best_threshold} (IoU: {best_iou:.4f})")

    if flood_ratio < 0.001:
        print("🔧 CRITICAL: Dataset has <0.1% flood pixels")
        print("   - Consider focal loss with higher alpha")
        print("   - Use weighted loss functions")
        print("   - Check if ground truth masks are correct")
    elif best_iou < 0.1:
        print("🔧 Model performance is very poor:")
        print("   - Train for more epochs")
        print("   - Reduce learning rate")
        print("   - Check data preprocessing")
        print("   - Verify model architecture")
    elif all_outputs.max() < 0.3:
        print("🔧 Model outputs are too low (max < 0.3):")
        print("   - Model might need more training")
        print("   - Check loss function scaling")
        print("   - Consider adjusting learning rate")

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Plot 1: Model output distribution
    plt.subplot(1, 3, 1)
    plt.hist(all_outputs.flatten().numpy(), bins=50, alpha=0.7, color="blue")
    plt.axvline(x=0.5, color="red", linestyle="--", label="Standard Threshold")
    plt.axvline(
        x=best_threshold,
        color="green",
        linestyle="--",
        label=f"Best Threshold ({best_threshold})",
    )
    plt.xlabel("Model Output (Probability)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Outputs")
    plt.legend()

    # Plot 2: Ground truth distribution
    plt.subplot(1, 3, 2)
    plt.hist(all_masks.flatten().numpy(), bins=50, alpha=0.7, color="orange")
    plt.xlabel("Ground Truth Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ground Truth")

    # Plot 3: IoU vs Threshold
    plt.subplot(1, 3, 3)
    ious = []
    for threshold in np.linspace(0.1, 0.9, 50):
        pred_binary = (all_outputs > threshold).float()
        tp = ((pred_binary == 1) & (all_masks == 1)).sum().item()
        fp = ((pred_binary == 1) & (all_masks == 0)).sum().item()
        fn = ((pred_binary == 0) & (all_masks == 1)).sum().item()
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou)

    plt.plot(np.linspace(0.1, 0.9, 50), ious, color="green")
    plt.axvline(
        x=best_threshold, color="red", linestyle="--", label=f"Best: {best_threshold}"
    )
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("IoU vs Threshold")
    plt.legend()

    plt.tight_layout()
    plt.savefig("model_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n📈 Analysis visualization saved as: model_analysis.png")


if __name__ == "__main__":
    analyze_dataset_and_model()
