import torch
import numpy as np
from torch.utils.data import DataLoader
from model import FloodSegmentationModel
from dataset import FloodDataset
import torch.nn.functional as F
import os


def quick_threshold_test():
    """Quick test to find the optimal threshold for current model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
    sar_dir = os.path.join(data_path, "S1Hand")
    optical_dir = os.path.join(data_path, "S2Hand")
    mask_dir = os.path.join(data_path, "LabelHand")

    dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Load model
    model = FloodSegmentationModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    print("Collecting predictions...")
    with torch.no_grad():
        for i, (sar, optical, masks) in enumerate(dataloader):
            if i >= 10:  # Limit for speed
                break

            sar, optical, masks = sar.to(device), optical.to(device), masks.to(device)

            outputs = model(sar, optical)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            predictions = torch.sigmoid(outputs)

            all_preds.append(predictions.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(f"Prediction statistics:")
    print(f"  Range: [{all_preds.min():.6f}, {all_preds.max():.6f}]")
    print(f"  Mean: {all_preds.mean():.6f}")
    print(f"  Std: {all_preds.std():.6f}")
    print(f"  25th percentile: {torch.quantile(all_preds, 0.25):.6f}")
    print(f"  50th percentile: {torch.quantile(all_preds, 0.50):.6f}")
    print(f"  75th percentile: {torch.quantile(all_preds, 0.75):.6f}")
    print(f"  90th percentile: {torch.quantile(all_preds, 0.90):.6f}")
    print(f"  95th percentile: {torch.quantile(all_preds, 0.95):.6f}")
    print(f"  99th percentile: {torch.quantile(all_preds, 0.99):.6f}")

    # Test a wide range of thresholds
    thresholds = np.arange(0.5, 1.0, 0.01)  # From 0.5 to 0.99

    print(f"\nTesting {len(thresholds)} thresholds...")
    print("Threshold | IoU    | F1     | Precision | Recall | Pred_Flood | True_Flood")
    print("-" * 75)

    best_iou = 0
    best_threshold = 0.5
    best_metrics = {}

    for threshold in thresholds:
        pred_binary = (all_preds > threshold).float()

        # Calculate metrics
        tp = ((pred_binary == 1) & (all_targets == 1)).sum().item()
        tn = ((pred_binary == 0) & (all_targets == 0)).sum().item()
        fp = ((pred_binary == 1) & (all_targets == 0)).sum().item()
        fn = ((pred_binary == 0) & (all_targets == 1)).sum().item()

        # Metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        pred_flood = pred_binary.sum().item()
        true_flood = all_targets.sum().item()

        # Print every 5th threshold for readability
        if len(str(threshold).split(".")[-1]) <= 2 or threshold in [
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
        ]:
            print(
                f"{threshold:9.2f} | {iou:6.4f} | {f1:6.4f} | {precision:9.4f} | {recall:6.4f} | {pred_flood:10.0f} | {true_flood:10.0f}"
            )

        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "pred_flood": pred_flood,
                "true_flood": true_flood,
            }

    print(f"\n🏆 BEST THRESHOLD: {best_threshold:.3f}")
    print(f"📊 BEST METRICS:")
    print(f"   IoU: {best_metrics['iou']:.4f}")
    print(f"   F1-Score: {best_metrics['f1']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   Predicted Flood Pixels: {best_metrics['pred_flood']:,.0f}")
    print(f"   True Flood Pixels: {best_metrics['true_flood']:,.0f}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if best_threshold > 0.9:
        print("   - Model is over-predicting severely (best threshold > 0.9)")
        print("   - Consider retraining with balanced loss function")
        print("   - Lower learning rate might help")
    elif best_threshold > 0.7:
        print("   - Model is over-predicting (best threshold > 0.7)")
        print("   - Consider adjusting loss function weights")
    else:
        print("   - Model threshold looks reasonable")

    if best_metrics["iou"] < 0.3:
        print(
            "   - IoU is still low, model needs more training or architecture changes"
        )

    return best_threshold, best_metrics


if __name__ == "__main__":
    quick_threshold_test()
