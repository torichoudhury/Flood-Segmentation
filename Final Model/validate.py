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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate comprehensive metrics for binary segmentation."""
    # Convert to binary predictions
    pred_binary = (predictions > threshold).float()
    target_binary = targets.float()

    # Flatten for metrics calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()

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

    return {
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
    }


def visualize_predictions(
    sar_images,
    optical_images,
    predictions,
    targets,
    save_dir="validation_results",
    num_samples=5,
):
    """Visualize model predictions alongside ground truth."""
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(num_samples, len(sar_images))):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Convert tensors to numpy for visualization
        sar_img = sar_images[i].squeeze().cpu().numpy()
        optical_img = optical_images[i].squeeze().cpu().numpy()
        pred_img = predictions[i].squeeze().cpu().numpy()
        target_img = targets[i].squeeze().cpu().numpy()

        # SAR Image
        axes[0, 0].imshow(sar_img, cmap="gray")
        axes[0, 0].set_title("SAR Image")
        axes[0, 0].axis("off")

        # Optical Image
        axes[0, 1].imshow(optical_img, cmap="gray")
        axes[0, 1].set_title("Optical Image")
        axes[0, 1].axis("off")

        # Ground Truth
        axes[0, 2].imshow(target_img, cmap="Blues")
        axes[0, 2].set_title("Ground Truth")
        axes[0, 2].axis("off")

        # Prediction (Raw)
        axes[1, 0].imshow(pred_img, cmap="Blues")
        axes[1, 0].set_title("Prediction (Raw)")
        axes[1, 0].axis("off")

        # Prediction (Binary)
        pred_binary = (pred_img > 0.5).astype(float)
        axes[1, 1].imshow(pred_binary, cmap="Blues")
        axes[1, 1].set_title("Prediction (Binary)")
        axes[1, 1].axis("off")

        # Difference (Error Map)
        diff = np.abs(pred_binary - target_img)
        axes[1, 2].imshow(diff, cmap="Reds")
        axes[1, 2].set_title("Error Map")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{i+1}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_confusion_matrix(predictions, targets, save_path="confusion_matrix.png"):
    """Plot confusion matrix."""
    pred_flat = (predictions > 0.5).view(-1).cpu().numpy()
    target_flat = targets.view(-1).cpu().numpy()

    cm = confusion_matrix(target_flat, pred_flat)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Flood", "Flood"],
        yticklabels=["No Flood", "Flood"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def validate_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth", help="Path to trained model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for validation"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="validation_results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset (you might want to create a separate validation dataset)
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")

    # For now, using the same dataset - ideally you'd have separate train/val splits
    val_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Load Model
    model = FloodSegmentationModel().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model file {args.model_path} not found! Using untrained model.")

    # Loss Functions & Metrics
    criterion = AdaptiveLoss().to(device)
    iou_metric = IoU().to(device)

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    all_predictions = []
    all_targets = []
    all_sar_images = []
    all_optical_images = []

    print("Starting validation...")
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

            # Resize outputs to match mask size if needed
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(outputs)

            # Calculate loss and IoU
            loss, _ = criterion(outputs, masks, 0)  # epoch=0 for validation
            iou = iou_metric(predictions, masks)

            val_loss += loss.item()
            val_iou += iou.item()

            # Store predictions for detailed analysis
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            all_sar_images.append(sar_images.cpu())
            all_optical_images.append(optical_images.cpu())

            # Update progress
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "iou": f"{iou.item():.4f}"}
            )

            # Limit samples for visualization (to avoid memory issues)
            if batch_idx >= 20:  # Only process first 20 batches for detailed analysis
                break

    # Calculate average metrics
    avg_loss = val_loss / len(progress_bar)
    avg_iou = val_iou / len(progress_bar)

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average IoU:  {avg_iou:.4f}")

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_sar_images = torch.cat(all_sar_images, dim=0)
    all_optical_images = torch.cat(all_optical_images, dim=0)

    # Calculate detailed metrics
    detailed_metrics = calculate_metrics(all_predictions, all_targets)

    print(f"\nDETAILED METRICS:")
    print(f"Accuracy:  {detailed_metrics['accuracy']:.4f}")
    print(f"Precision: {detailed_metrics['precision']:.4f}")
    print(f"Recall:    {detailed_metrics['recall']:.4f}")
    print(f"F1-Score:  {detailed_metrics['f1_score']:.4f}")
    print(f"IoU:       {detailed_metrics['iou']:.4f}")
    print(f"Dice:      {detailed_metrics['dice']:.4f}")

    print(f"\nCONFUSION MATRIX:")
    print(f"True Positives:  {detailed_metrics['tp']}")
    print(f"True Negatives:  {detailed_metrics['tn']}")
    print(f"False Positives: {detailed_metrics['fp']}")
    print(f"False Negatives: {detailed_metrics['fn']}")

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save visualizations
    print(f"\nGenerating visualizations...")
    visualize_predictions(
        all_sar_images,
        all_optical_images,
        all_predictions,
        all_targets,
        save_dir=args.save_dir,
        num_samples=10,
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        all_predictions,
        all_targets,
        save_path=os.path.join(args.save_dir, "confusion_matrix.png"),
    )

    # Save metrics to file
    metrics_file = os.path.join(args.save_dir, "validation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"VALIDATION RESULTS\n")
        f.write(f"{'='*50}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Average IoU:  {avg_iou:.4f}\n")
        f.write(f"\nDETAILED METRICS:\n")
        f.write(f"Accuracy:  {detailed_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {detailed_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {detailed_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {detailed_metrics['f1_score']:.4f}\n")
        f.write(f"IoU:       {detailed_metrics['iou']:.4f}\n")
        f.write(f"Dice:      {detailed_metrics['dice']:.4f}\n")
        f.write(f"\nCONFUSION MATRIX:\n")
        f.write(f"True Positives:  {detailed_metrics['tp']}\n")
        f.write(f"True Negatives:  {detailed_metrics['tn']}\n")
        f.write(f"False Positives: {detailed_metrics['fp']}\n")
        f.write(f"False Negatives: {detailed_metrics['fn']}\n")

    print(f"\nValidation complete! Results saved to: {args.save_dir}")
    print(f"- Sample visualizations: {args.save_dir}/sample_*.png")
    print(f"- Confusion matrix: {args.save_dir}/confusion_matrix.png")
    print(f"- Metrics summary: {args.save_dir}/validation_metrics.txt")


if __name__ == "__main__":
    validate_model()
