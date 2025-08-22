import torch
import numpy as np
from model import FloodSegmentationModel
from dataset import FloodDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def calculate_metrics_at_thresholds(
    preds, targets, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):
    """Calculate detailed metrics at different thresholds"""
    results = {}

    for thresh in thresholds:
        # Convert to binary predictions
        pred_binary = (preds > thresh).float()

        # Calculate metrics
        tp = ((pred_binary == 1) & (targets == 1)).sum().item()
        tn = ((pred_binary == 0) & (targets == 0)).sum().item()
        fp = ((pred_binary == 1) & (targets == 0)).sum().item()
        fn = ((pred_binary == 0) & (targets == 1)).sum().item()

        # Calculate metrics with epsilon to avoid division by zero
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        # Count predicted flood pixels
        pred_flood_count = (pred_binary == 1).sum().item()

        results[thresh] = {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "pred_flood": pred_flood_count,
        }

    return results


def validate_model(model_path="best_model_improved.pth"):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FloodSegmentationModel().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load {model_path}, using randomly initialized model")

    model.eval()

    # Load dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = data_path + "/S1Hand"
    optical_dir = data_path + "/S2Hand"
    mask_dir = data_path + "/LabelHand"

    dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_targets = []

    print("Running validation...")

    with torch.no_grad():
        for sar, optical, mask in loader:
            sar, optical, mask = sar.to(device), optical.to(device), mask.to(device)

            # Forward pass
            output = model(sar, optical)

            # Resize if needed
            if output.shape[-2:] != mask.shape[-2:]:
                output = F.interpolate(
                    output, size=mask.shape[-2:], mode="bilinear", align_corners=False
                )

            # Apply sigmoid
            output = torch.sigmoid(output)

            # Collect predictions and targets
            all_preds.append(output.cpu())
            all_targets.append(mask.cpu())

    # Concatenate all results
    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_targets = torch.cat(all_targets, dim=0).flatten()

    print(f"Validation completed. Total pixels: {len(all_preds)}")
    print(
        f"Ground truth flood pixels: {(all_targets == 1).sum().item()} ({100*(all_targets == 1).float().mean():.2f}%)"
    )

    # Calculate metrics at different thresholds
    results = calculate_metrics_at_thresholds(all_preds, all_targets)

    # Print results in the same format as before
    print("\nMetrics by Threshold:")
    print("Thresh |    IoU |     F1 |   Prec |    Rec | Pred Flood")
    print("------------------------------------------------------------")

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics = results[thresh]
        print(
            f"  {thresh:4.1f} | {metrics['iou']:6.4f} | {metrics['f1']:6.4f} | {metrics['precision']:6.4f} | {metrics['recall']:6.4f} | {metrics['pred_flood']:10,d}"
        )


if __name__ == "__main__":
    validate_model()
