import torch
import torch.nn.functional as F
from model import FloodSegmentationModel
from dataset import FloodDataset
from torch.utils.data import DataLoader
import os

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
data_path = "D:/Flood-Segmentation/dataset/HandLabeled"
sar_dir = os.path.join(data_path, "S1Hand")
optical_dir = os.path.join(data_path, "S2Hand")
mask_dir = os.path.join(data_path, "LabelHand")
train_dataset = FloodDataset(sar_dir, optical_dir, mask_dir)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

# Model Initialization
model = FloodSegmentationModel().to(device)

# Test one batch
for sar_images, optical_images, masks in train_loader:
    sar_images, optical_images, masks = (
        sar_images.to(device),
        optical_images.to(device),
        masks.to(device),
    )

    print(f"Input SAR shape: {sar_images.shape}")
    print(f"Input Optical shape: {optical_images.shape}")
    print(f"Input Mask shape: {masks.shape}")

    with torch.no_grad():
        outputs = model(sar_images, optical_images)
        print(f"Model output shape: {outputs.shape}")

        # Resize outputs to match mask size if needed
        if outputs.shape[-2:] != masks.shape[-2:]:
            print("Resizing outputs to match mask size...")
            outputs_resized = F.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            print(f"Resized output shape: {outputs_resized.shape}")
        else:
            print("Output and mask sizes already match")

    break
