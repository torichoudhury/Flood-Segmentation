import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import FloodSegmentationModel
from loss import AdaptiveLoss, IoU
from dataset import FloodDataset
from tqdm import tqdm


def validate_with_optimal_threshold():
    """Validate model using the optimal threshold found (0.99)."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth", help="Path to trained model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for validation"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.99, help="Optimal threshold"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="validation_optimized",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using optimal threshold: {args.threshold}")

    # Load Dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")

    val_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Load Model
    model = FloodSegmentationModel().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✅ Loaded model from {args.model_path}")
    else:
        print(f"❌ Model file {args.model_path} not found!")
        return

    # Custom IoU with optimal threshold
    iou_metric = IoU(threshold=args.threshold).to(device)

    # Validation
    model.eval()
    val_iou = 0.0
    all_predictions = []
    all_targets = []

    print("Starting optimized validation...")
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=True)

        for batch_idx, (sar_images, optical_images, masks) in enumerate(progress_bar):
            sar_images, optical_images, masks = (
                sar_images.to(device),
                optical_images.to(device),
                masks.to(device),
            )

            # Forward pass
            outputs = model(sar_images, optical_images)

            # Resize outputs to match mask size
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(outputs)

            # Calculate IoU with optimal threshold
            iou = iou_metric(outputs, masks)  # iou_metric applies sigmoid internally
            val_iou += iou.item()

            # Store for detailed analysis (limit to avoid memory issues)
            if batch_idx < 10:
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())

            # Update progress
            progress_bar.set_postfix(
                {
                    "iou": f"{iou.item():.4f}",
                    "pred_range": f"[{predictions.min():.3f},{predictions.max():.3f}]",
                }
            )

            if batch_idx >= 50:  # Limit for demonstration
                break

    # Calculate detailed metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Apply optimal threshold
    pred_binary = (all_predictions > args.threshold).float()

    # Calculate confusion matrix
    tp = ((pred_binary == 1) & (all_targets == 1)).sum().item()
    tn = ((pred_binary == 0) & (all_targets == 0)).sum().item()
    fp = ((pred_binary == 1) & (all_targets == 0)).sum().item()
    fn = ((pred_binary == 0) & (all_targets == 1)).sum().item()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou_final = tp / (tp + fp + fn + 1e-8)

    avg_iou = val_iou / min(len(progress_bar), 51)

    print(f"\n{'='*60}")
    print(f"OPTIMIZED VALIDATION RESULTS (Threshold: {args.threshold})")
    print(f"{'='*60}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"\nDETAILED METRICS:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU:       {iou_final:.4f}")

    print(f"\nCONFUSION MATRIX:")
    print(f"True Positives:  {tp:,.0f}")
    print(f"True Negatives:  {tn:,.0f}")
    print(f"False Positives: {fp:,.0f}")
    print(f"False Negatives: {fn:,.0f}")

    # Improvement comparison
    print(f"\n📈 IMPROVEMENT COMPARISON:")
    print(f"Previous (threshold 0.5): IoU ~0.077, Precision ~0.077")
    print(
        f"Current (threshold {args.threshold}): IoU {iou_final:.4f}, Precision {precision:.4f}"
    )
    improvement = ((precision - 0.077) / 0.077) * 100
    print(f"Precision improvement: {improvement:+.1f}%")

    # Create results directory and save
    os.makedirs(args.save_dir, exist_ok=True)

    results_file = os.path.join(args.save_dir, "optimized_validation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"OPTIMIZED VALIDATION RESULTS (Threshold: {args.threshold})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"\nDETAILED METRICS:\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"IoU:       {iou_final:.4f}\n")
        f.write(f"\nCONFUSION MATRIX:\n")
        f.write(f"True Positives:  {tp:,.0f}\n")
        f.write(f"True Negatives:  {tn:,.0f}\n")
        f.write(f"False Positives: {fp:,.0f}\n")
        f.write(f"False Negatives: {fn:,.0f}\n")

    print(f"\n✅ Results saved to: {results_file}")


# Custom IoU class with adjustable threshold
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


if __name__ == "__main__":
    validate_with_optimal_threshold()
