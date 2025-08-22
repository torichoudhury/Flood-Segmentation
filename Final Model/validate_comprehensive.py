import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import FloodSegmentationModel
from dataset import FloodDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def calculate_metrics_with_thresholds(predictions, targets, thresholds=[0.3, 0.5, 0.7]):
    """Calculate metrics for multiple thresholds."""
    results = {}

    for threshold in thresholds:
        # Convert to binary predictions
        pred_binary = (predictions > threshold).float()
        target_binary = targets.float()

        # Basic metrics
        tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
        tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-8)

        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        results[threshold] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "iou": iou,
            "dice": dice,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "predicted_flood_pixels": pred_binary.sum().item(),
            "actual_flood_pixels": target_binary.sum().item(),
        }

    return results


def create_detailed_visualization(
    sar_images,
    optical_images,
    predictions,
    targets,
    save_dir="validation_results",
    num_samples=3,
):
    """Create detailed visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(num_samples, len(sar_images))):
        # Convert tensors to numpy
        sar_img = sar_images[i].squeeze().cpu().numpy()
        optical_img = optical_images[i].squeeze().cpu().numpy()
        pred_img = predictions[i].squeeze().cpu().numpy()
        target_img = targets[i].squeeze().cpu().numpy()

        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # Row 1: Input images
        axes[0, 0].imshow(sar_img, cmap="gray")
        axes[0, 0].set_title("SAR Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(optical_img, cmap="gray")
        axes[0, 1].set_title("Optical Image")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(target_img, cmap="Blues", vmin=0, vmax=1)
        axes[0, 2].set_title(f"Ground Truth\n({target_img.sum():.0f} flood pixels)")
        axes[0, 2].axis("off")

        # Prediction statistics
        pred_stats = f"Range: [{pred_img.min():.3f}, {pred_img.max():.3f}]\nMean: {pred_img.mean():.3f}"
        axes[0, 3].text(
            0.1, 0.5, pred_stats, fontsize=12, transform=axes[0, 3].transAxes
        )
        axes[0, 3].set_title("Prediction Stats")
        axes[0, 3].axis("off")

        # Row 2: Predictions at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        for j, threshold in enumerate(thresholds):
            pred_binary = (pred_img > threshold).astype(float)
            axes[1, j].imshow(pred_binary, cmap="Blues", vmin=0, vmax=1)
            axes[1, j].set_title(
                f"Prediction > {threshold}\n({pred_binary.sum():.0f} flood pixels)"
            )
            axes[1, j].axis("off")

        # Raw prediction heatmap
        axes[1, 3].imshow(pred_img, cmap="Blues", vmin=0, vmax=1)
        axes[1, 3].set_title("Raw Prediction\n(Probability)")
        axes[1, 3].axis("off")

        # Row 3: Error analysis
        for j, threshold in enumerate(thresholds):
            pred_binary = (pred_img > threshold).astype(float)

            # Create error map: TP=green, TN=black, FP=red, FN=yellow
            error_map = np.zeros((*pred_img.shape, 3))  # RGB

            tp_mask = (pred_binary == 1) & (target_img == 1)  # True Positive
            tn_mask = (pred_binary == 0) & (target_img == 0)  # True Negative
            fp_mask = (pred_binary == 1) & (target_img == 0)  # False Positive
            fn_mask = (pred_binary == 0) & (target_img == 1)  # False Negative

            error_map[tp_mask] = [0, 1, 0]  # Green for TP
            error_map[tn_mask] = [0, 0, 0]  # Black for TN
            error_map[fp_mask] = [1, 0, 0]  # Red for FP
            error_map[fn_mask] = [1, 1, 0]  # Yellow for FN

            axes[2, j].imshow(error_map)
            axes[2, j].set_title(
                f"Error Map (t={threshold})\nTP:{tp_mask.sum()}, FP:{fp_mask.sum()}, FN:{fn_mask.sum()}"
            )
            axes[2, j].axis("off")

        # Legend
        legend_text = "Error Map Legend:\nGreen = True Positive\nRed = False Positive\nYellow = False Negative\nBlack = True Negative"
        axes[2, 3].text(
            0.1,
            0.5,
            legend_text,
            fontsize=10,
            transform=axes[2, 3].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        axes[2, 3].set_title("Legend")
        axes[2, 3].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/detailed_sample_{i+1}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def validate_model_comprehensive():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="best_model_improved.pth", help="Path to trained model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for validation"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="validation_results_new",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        print(f"❌ Model file {args.model_path} not found! Using untrained model.")

    # Validation
    model.eval()
    all_predictions = []
    all_targets = []
    all_sar_images = []
    all_optical_images = []

    print("\n🔍 Running comprehensive validation...")
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=True)

        for batch_idx, (sar_images, optical_images, masks) in enumerate(progress_bar):
            sar_images, optical_images, masks = (
                sar_images.to(device),
                optical_images.to(device),
                masks.to(device),
            )

            # Forward pass
            outputs = model(sar_images, optical_images)  # Single output now

            # Resize outputs to match mask size
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(outputs)

            # Store for analysis (limit to prevent memory issues)
            if batch_idx < 20:
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())
                all_sar_images.append(sar_images.cpu())
                all_optical_images.append(optical_images.cpu())

            # Show progress
            pred_stats = f"pred_range:[{predictions.min():.3f},{predictions.max():.3f}]"
            mask_flood = masks.sum().item()
            progress_bar.set_postfix_str(f"{pred_stats}, flood_pixels:{mask_flood:.0f}")

            if batch_idx >= 25:  # Limit for memory
                break

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_sar_images = torch.cat(all_sar_images, dim=0)
    all_optical_images = torch.cat(all_optical_images, dim=0)

    # Calculate metrics for multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics_results = calculate_metrics_with_thresholds(
        all_predictions, all_targets, thresholds
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE VALIDATION RESULTS")
    print(f"{'='*80}")

    print(f"\nModel Performance Summary:")
    print(
        f"Prediction Range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]"
    )
    print(f"Prediction Mean: {all_predictions.mean():.6f}")
    print(f"Total Ground Truth Flood Pixels: {all_targets.sum().item():,.0f}")

    print(f"\nMetrics by Threshold:")
    print(
        f"{'Thresh':>6} | {'IoU':>6} | {'F1':>6} | {'Prec':>6} | {'Rec':>6} | {'Pred Flood':>10}"
    )
    print(f"{'-'*60}")

    best_iou = 0
    best_threshold = 0.5

    for threshold in thresholds:
        metrics = metrics_results[threshold]
        print(
            f"{threshold:6.1f} | {metrics['iou']:6.4f} | {metrics['f1_score']:6.4f} | "
            f"{metrics['precision']:6.4f} | {metrics['recall']:6.4f} | {metrics['predicted_flood_pixels']:10,.0f}"
        )

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_threshold = threshold

    print(f"\n🏆 Best Threshold: {best_threshold} (IoU: {best_iou:.4f})")

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate detailed visualizations
    print(f"\n📊 Generating detailed visualizations...")
    create_detailed_visualization(
        all_sar_images,
        all_optical_images,
        all_predictions,
        all_targets,
        save_dir=args.save_dir,
        num_samples=5,
    )

    # Save comprehensive results
    results_file = os.path.join(args.save_dir, "comprehensive_validation.txt")
    with open(results_file, "w") as f:
        f.write(f"COMPREHENSIVE VALIDATION RESULTS\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model Performance Summary:\n")
        f.write(
            f"Prediction Range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]\n"
        )
        f.write(f"Prediction Mean: {all_predictions.mean():.6f}\n")
        f.write(f"Total Ground Truth Flood Pixels: {all_targets.sum().item():,.0f}\n\n")
        f.write(f"Best Threshold: {best_threshold} (IoU: {best_iou:.4f})\n\n")
        f.write(f"Detailed Metrics by Threshold:\n")
        f.write(f"{'='*50}\n")

        for threshold in thresholds:
            metrics = metrics_results[threshold]
            f.write(f"\nThreshold: {threshold}\n")
            f.write(f"  IoU: {metrics['iou']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(
                f"  Predicted Flood Pixels: {metrics['predicted_flood_pixels']:,.0f}\n"
            )
            f.write(f"  True Positives: {metrics['tp']:,.0f}\n")
            f.write(f"  False Positives: {metrics['fp']:,.0f}\n")
            f.write(f"  False Negatives: {metrics['fn']:,.0f}\n")
            f.write(f"  True Negatives: {metrics['tn']:,.0f}\n")

    print(f"\n✅ Comprehensive validation complete!")
    print(f"📁 Results saved to: {args.save_dir}/")
    print(f"📄 Detailed report: {results_file}")
    print(f"🖼️ Visualizations: {args.save_dir}/detailed_sample_*.png")


if __name__ == "__main__":
    validate_model_comprehensive()
