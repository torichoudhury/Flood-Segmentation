"""
Quick test script to debug the training issues
"""

import torch
import torch.nn.functional as F
from dataset import FloodDataset
from torch.utils.data import DataLoader, random_split


def test_dataset_loading():
    print("Testing dataset loading...")
    try:
        dataset = FloodDataset(
            "D:/Flood-Segmentation/dataset/HandLabeled/S1Hand",
            "D:/Flood-Segmentation/dataset/HandLabeled/S2Hand",
            "D:/Flood-Segmentation/dataset/HandLabeled/LabelHand",
        )
        print(f"Dataset length: {len(dataset)}")

        # Test loading first item
        sar, opt, mask = dataset[0]
        print(f"SAR shape: {sar.shape}")
        print(f"Optical shape: {opt.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
        print(f"Mask mean: {mask.mean():.4f}")

        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        sar_batch, opt_batch, mask_batch = batch
        print(f"Batch SAR shape: {sar_batch.shape}")
        print(f"Batch Optical shape: {opt_batch.shape}")
        print(f"Batch Mask shape: {mask_batch.shape}")
        print(f"Batch mask unique values: {torch.unique(mask_batch)}")

        return True
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return False


def test_class_distribution():
    print("\nTesting class distribution...")
    try:
        dataset = FloodDataset(
            "D:/Flood-Segmentation/dataset/HandLabeled/S1Hand",
            "D:/Flood-Segmentation/dataset/HandLabeled/S2Hand",
            "D:/Flood-Segmentation/dataset/HandLabeled/LabelHand",
        )

        total_pixels = 0
        flood_pixels = 0

        # Sample first 10 items to check class distribution
        for i in range(min(10, len(dataset))):
            _, _, mask = dataset[i]
            total_pixels += mask.numel()
            flood_pixels += (mask > 0.5).sum().item()

        flood_ratio = flood_pixels / total_pixels
        print(f"Flood pixel ratio: {flood_ratio:.4f} ({flood_ratio*100:.2f}%)")
        print(
            f"Background pixel ratio: {1-flood_ratio:.4f} ({(1-flood_ratio)*100:.2f}%)"
        )

        # Suggest better class weights
        if flood_ratio < 0.1:
            suggested_weight = min(10.0, 1.0 / flood_ratio * 0.1)
            print(f"Suggested flood class weight: {suggested_weight:.1f}")

        return flood_ratio
    except Exception as e:
        print(f"Class distribution test failed: {e}")
        return None


if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        flood_ratio = test_class_distribution()
        print(f"\nDataset tests completed successfully!")
        if flood_ratio and flood_ratio < 0.05:
            print(
                "⚠️  Warning: Very low flood ratio detected. Consider using stronger class weighting."
            )
    else:
        print("❌ Dataset tests failed!")
