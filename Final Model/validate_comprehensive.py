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
import json
from datetime import datetime


def create_publication_confusion_matrix(
    predictions, targets, threshold=0.4, save_path="confusion_matrix.png"
):
    """Create a publication-ready confusion matrix."""
    # Convert to binary predictions
    pred_binary = (predictions > threshold).cpu().numpy().flatten()
    target_binary = targets.cpu().numpy().flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(target_binary, pred_binary)

    # Calculate normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Raw confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=["Non-Flood", "Flood"],
        yticklabels=["Non-Flood", "Flood"],
    )
    ax1.set_title(
        f"Confusion Matrix (Raw Counts)\nThreshold: {threshold}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)

    # Normalized confusion matrix
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=ax2,
        xticklabels=["Non-Flood", "Flood"],
        yticklabels=["Non-Flood", "Flood"],
    )
    ax2.set_title(
        f"Normalized Confusion Matrix\nThreshold: {threshold}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Predicted Label", fontsize=12)
    ax2.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # Calculate detailed metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return metrics, cm


def create_threshold_analysis_plot(metrics_results, save_path="threshold_analysis.png"):
    """Create threshold analysis plot for paper."""
    thresholds = list(metrics_results.keys())

    # Extract metrics
    ious = [metrics_results[t]["iou"] for t in thresholds]
    f1_scores = [metrics_results[t]["f1_score"] for t in thresholds]
    precisions = [metrics_results[t]["precision"] for t in thresholds]
    recalls = [metrics_results[t]["recall"] for t in thresholds]

    # Create the plot
    plt.figure(figsize=(12, 8))

    plt.plot(thresholds, ious, "b-o", linewidth=2, markersize=6, label="IoU")
    plt.plot(thresholds, f1_scores, "r-s", linewidth=2, markersize=6, label="F1-Score")
    plt.plot(
        thresholds, precisions, "g-^", linewidth=2, markersize=6, label="Precision"
    )
    plt.plot(thresholds, recalls, "m-v", linewidth=2, markersize=6, label="Recall")

    # Find best threshold
    best_iou_idx = np.argmax(ious)
    best_threshold = thresholds[best_iou_idx]
    best_iou = ious[best_iou_idx]

    # Mark best threshold
    plt.axvline(x=best_threshold, color="red", linestyle="--", alpha=0.7, linewidth=2)
    plt.text(
        best_threshold + 0.02,
        best_iou - 0.05,
        f"Best Threshold\n{best_threshold} (IoU: {best_iou:.3f})",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        fontsize=10,
    )

    plt.xlabel("Threshold", fontsize=14, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=14, fontweight="bold")
    plt.title(
        "Model Performance vs. Classification Threshold", fontsize=16, fontweight="bold"
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds) - 0.01, max(thresholds) + 0.01)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_metrics_comparison_plot(metrics_results, save_path="metrics_comparison.png"):
    """Create a comprehensive metrics comparison plot."""
    thresholds = list(metrics_results.keys())

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # IoU and F1-Score
    ious = [metrics_results[t]["iou"] for t in thresholds]
    f1_scores = [metrics_results[t]["f1_score"] for t in thresholds]

    ax1.plot(thresholds, ious, "b-o", linewidth=2, markersize=6, label="IoU")
    ax1.plot(thresholds, f1_scores, "r-s", linewidth=2, markersize=6, label="F1-Score")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("IoU and F1-Score vs Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision and Recall
    precisions = [metrics_results[t]["precision"] for t in thresholds]
    recalls = [metrics_results[t]["recall"] for t in thresholds]

    ax2.plot(
        thresholds, precisions, "g-^", linewidth=2, markersize=6, label="Precision"
    )
    ax2.plot(thresholds, recalls, "m-v", linewidth=2, markersize=6, label="Recall")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision and Recall vs Threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Dice coefficient
    dice_scores = [metrics_results[t]["dice"] for t in thresholds]

    ax3.plot(
        thresholds,
        dice_scores,
        "c-d",
        linewidth=2,
        markersize=6,
        label="Dice Coefficient",
    )
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Score")
    ax3.set_title("Dice Coefficient vs Threshold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Predicted vs Actual flood pixels
    predicted_pixels = [
        metrics_results[t]["predicted_flood_pixels"] for t in thresholds
    ]
    actual_pixels = metrics_results[thresholds[0]][
        "actual_flood_pixels"
    ]  # Same for all thresholds

    ax4.plot(
        thresholds,
        predicted_pixels,
        "orange",
        linewidth=2,
        markersize=6,
        label="Predicted",
    )
    ax4.axhline(
        y=actual_pixels, color="red", linestyle="--", linewidth=2, label="Actual"
    )
    ax4.set_xlabel("Threshold")
    ax4.set_ylabel("Flood Pixels")
    ax4.set_title("Predicted vs Actual Flood Pixels")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_training_validation_plots(
    checkpoint_path, save_path="training_validation_curves.png"
):
    """Create training vs validation plots if checkpoint contains history."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Check if checkpoint contains training history
        if "training_history" not in checkpoint:
            print(f"⚠️ No training history found in {checkpoint_path}")
            return None

        history = checkpoint["training_history"]

        # Create the plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss
        axes[0, 0].plot(
            epochs, history["train_loss"], "b-", linewidth=2, label="Training"
        )
        if "val_loss" in history:
            axes[0, 0].plot(
                epochs, history["val_loss"], "r-", linewidth=2, label="Validation"
            )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training vs Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1 Score
        axes[0, 1].plot(
            epochs, history["train_f1"], "b-", linewidth=2, label="Training"
        )
        if "val_f1" in history:
            axes[0, 1].plot(
                epochs, history["val_f1"], "r-", linewidth=2, label="Validation"
            )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("Training vs Validation F1 Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # IoU
        axes[0, 2].plot(
            epochs, history["train_iou"], "b-", linewidth=2, label="Training"
        )
        if "val_iou" in history:
            axes[0, 2].plot(
                epochs, history["val_iou"], "r-", linewidth=2, label="Validation"
            )
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("IoU")
        axes[0, 2].set_title("Training vs Validation IoU")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Precision
        axes[1, 0].plot(
            epochs, history["train_precision"], "b-", linewidth=2, label="Training"
        )
        if "val_precision" in history:
            axes[1, 0].plot(
                epochs, history["val_precision"], "r-", linewidth=2, label="Validation"
            )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Training vs Validation Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall
        axes[1, 1].plot(
            epochs, history["train_recall"], "b-", linewidth=2, label="Training"
        )
        if "val_recall" in history:
            axes[1, 1].plot(
                epochs, history["val_recall"], "r-", linewidth=2, label="Validation"
            )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].set_title("Training vs Validation Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Learning Rate
        if "learning_rate" in history:
            axes[1, 2].plot(epochs, history["learning_rate"], "g-", linewidth=2)
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Learning Rate")
            axes[1, 2].set_title("Learning Rate Schedule")
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_yscale("log")
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "Learning Rate\nHistory Not Available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].set_title("Learning Rate Schedule")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Training/validation plots saved to {save_path}")
        return save_path

    except Exception as e:
        print(f"❌ Error creating training/validation plots: {e}")
        return None


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
        "--model_path",
        type=str,
        default="best_model_optimized.pth",
        help="Path to trained model",
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

    # Generate publication-ready visualizations
    print(f"\n📊 Generating publication-ready visualizations...")

    # 1. Confusion Matrix
    print("Creating confusion matrix...")
    cm_metrics, cm_data = create_publication_confusion_matrix(
        all_predictions,
        all_targets,
        threshold=best_threshold,
        save_path=os.path.join(args.save_dir, "confusion_matrix.png"),
    )

    # 2. Threshold Analysis Plot
    print("Creating threshold analysis plot...")
    create_threshold_analysis_plot(
        metrics_results, save_path=os.path.join(args.save_dir, "threshold_analysis.png")
    )

    # 3. Comprehensive Metrics Comparison
    print("Creating metrics comparison plot...")
    create_metrics_comparison_plot(
        metrics_results, save_path=os.path.join(args.save_dir, "metrics_comparison.png")
    )

    # 4. Training/Validation Curves (if available)
    print("Checking for training history...")
    checkpoint_files = ["checkpoint_optimized.pth", "checkpoint_improved.pth"]
    training_plots_created = False

    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            result = create_training_validation_plots(
                checkpoint_file,
                save_path=os.path.join(
                    args.save_dir,
                    f"training_curves_{checkpoint_file.replace('.pth', '')}.png",
                ),
            )
            if result:
                training_plots_created = True
                break

    if not training_plots_created:
        print(
            "⚠️ No training history found in checkpoints. Consider running enhanced training script."
        )

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
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model Performance Summary:\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(
            f"Prediction Range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]\n"
        )
        f.write(f"Prediction Mean: {all_predictions.mean():.6f}\n")
        f.write(f"Total Ground Truth Flood Pixels: {all_targets.sum().item():,.0f}\n")
        f.write(f"Total Samples Evaluated: {len(all_predictions)}\n\n")
        f.write(f"Best Threshold: {best_threshold} (IoU: {best_iou:.4f})\n\n")

        # Best threshold detailed metrics
        best_metrics = metrics_results[best_threshold]
        f.write(f"BEST THRESHOLD PERFORMANCE ({best_threshold}):\n")
        f.write(f"{'='*40}\n")
        f.write(f"IoU (Jaccard Index): {best_metrics['iou']:.4f}\n")
        f.write(f"F1-Score: {best_metrics['f1_score']:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall (Sensitivity): {best_metrics['recall']:.4f}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"Dice Coefficient: {best_metrics['dice']:.4f}\n")
        f.write(f"True Positives: {best_metrics['tp']:,.0f}\n")
        f.write(f"False Positives: {best_metrics['fp']:,.0f}\n")
        f.write(f"False Negatives: {best_metrics['fn']:,.0f}\n")
        f.write(f"True Negatives: {best_metrics['tn']:,.0f}\n")
        f.write(
            f"Predicted Flood Pixels: {best_metrics['predicted_flood_pixels']:,.0f}\n\n"
        )

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
            f.write(f"  Dice Coefficient: {metrics['dice']:.4f}\n")
            f.write(
                f"  Predicted Flood Pixels: {metrics['predicted_flood_pixels']:,.0f}\n"
            )
            f.write(f"  True Positives: {metrics['tp']:,.0f}\n")
            f.write(f"  False Positives: {metrics['fp']:,.0f}\n")
            f.write(f"  False Negatives: {metrics['fn']:,.0f}\n")
            f.write(f"  True Negatives: {metrics['tn']:,.0f}\n")

    # Save metrics as JSON for further analysis
    json_results = {
        "model_path": args.model_path,
        "evaluation_timestamp": datetime.now().isoformat(),
        "prediction_stats": {
            "min": float(all_predictions.min()),
            "max": float(all_predictions.max()),
            "mean": float(all_predictions.mean()),
            "std": float(all_predictions.std()),
        },
        "total_samples": len(all_predictions),
        "total_ground_truth_flood_pixels": int(all_targets.sum().item()),
        "best_threshold": float(best_threshold),
        "best_iou": float(best_iou),
        "metrics_by_threshold": {
            str(t): {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            }
            for t, metrics in metrics_results.items()
        },
    }

    json_file = os.path.join(args.save_dir, "validation_metrics.json")
    with open(json_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✅ Comprehensive validation complete!")
    print(f"📁 Results saved to: {args.save_dir}/")
    print(f"📄 Detailed report: {results_file}")
    print(f"📊 JSON metrics: {json_file}")
    print(f"🖼️ Sample visualizations: {args.save_dir}/detailed_sample_*.png")
    print(f"📈 Publication plots:")
    print(f"    - Confusion Matrix: {args.save_dir}/confusion_matrix.png")
    print(f"    - Threshold Analysis: {args.save_dir}/threshold_analysis.png")
    print(f"    - Metrics Comparison: {args.save_dir}/metrics_comparison.png")
    if training_plots_created:
        print(f"    - Training Curves: {args.save_dir}/training_curves_*.png")


if __name__ == "__main__":
    validate_model_comprehensive()
