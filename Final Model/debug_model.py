import torch
import numpy as np
from model import FloodSegmentationModel
from dataset import FloodDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load model and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FloodSegmentationModel().to(device)

data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
sar_dir = data_path + "/S1Hand"
optical_dir = data_path + "/S2Hand"
mask_dir = data_path + "/LabelHand"

dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Get one sample
sar, optical, mask = next(iter(loader))
sar, optical, mask = sar.to(device), optical.to(device), mask.to(device)

print("Input shapes:")
print(f"SAR: {sar.shape}, range: [{sar.min():.3f}, {sar.max():.3f}]")
print(f"Optical: {optical.shape}, range: [{optical.min():.3f}, {optical.max():.3f}]")
print(f"Mask: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")

# Forward pass
model.eval()
with torch.no_grad():
    output = model(sar, optical)

print(f"\nModel output shape: {output.shape}")
print(f"Model output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"Model output mean: {output.mean():.3f}")
print(f"Model output std: {output.std():.3f}")

# Apply sigmoid
sigmoid_output = torch.sigmoid(output)
print(f"After sigmoid range: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")
print(f"After sigmoid mean: {sigmoid_output.mean():.3f}")

# Check class distribution in mask
mask_flat = mask.flatten()
flood_pixels = (mask_flat == 1).sum().item()
total_pixels = mask_flat.numel()
print(f"\nGround truth statistics:")
print(f"Total pixels: {total_pixels}")
print(f"Flood pixels: {flood_pixels} ({100*flood_pixels/total_pixels:.2f}%)")
print(
    f"Non-flood pixels: {total_pixels-flood_pixels} ({100*(total_pixels-flood_pixels)/total_pixels:.2f}%)"
)

# Check prediction distribution at different thresholds
for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred = (sigmoid_output > thresh).float()
    pred_flood = (pred == 1).sum().item()
    print(
        f"At threshold {thresh}: {pred_flood} flood pixels ({100*pred_flood/total_pixels:.2f}%)"
    )
