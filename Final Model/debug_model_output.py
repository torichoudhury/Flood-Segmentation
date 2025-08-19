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
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

# Model Initialization
model = FloodSegmentationModel().to(device)

# Test one batch without autocast
for sar_images, optical_images, masks in train_loader:
    sar_images, optical_images, masks = (
        sar_images.to(device),
        optical_images.to(device),
        masks.to(device),
    )

    print(
        f"Input shapes - SAR: {sar_images.shape}, Optical: {optical_images.shape}, Mask: {masks.shape}"
    )

    # Check input ranges
    print(f"SAR range: [{sar_images.min()}, {sar_images.max()}]")
    print(f"Optical range: [{optical_images.min()}, {optical_images.max()}]")
    print(f"Mask range: [{masks.min()}, {masks.max()}]")

    with torch.no_grad():
        # Get model outputs without autocast
        outputs = model(sar_images, optical_images)
        print(f"Raw model output shape: {outputs.shape}")
        print(f"Raw model output range: [{outputs.min()}, {outputs.max()}]")
        print(f"Raw model output has NaN: {torch.isnan(outputs).any()}")
        print(f"Raw model output has Inf: {torch.isinf(outputs).any()}")

        # Apply sigmoid
        outputs_sigmoid = torch.sigmoid(outputs)
        print(
            f"Sigmoid output range: [{outputs_sigmoid.min()}, {outputs_sigmoid.max()}]"
        )
        print(f"Sigmoid output has NaN: {torch.isnan(outputs_sigmoid).any()}")

        # Resize outputs to match mask size if needed
        if outputs.shape[-2:] != masks.shape[-2:]:
            print("Resizing outputs to match mask size...")
            outputs_resized = F.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            print(f"Resized output shape: {outputs_resized.shape}")
            print(
                f"Resized output range: [{outputs_resized.min()}, {outputs_resized.max()}]"
            )
            print(f"Resized output has NaN: {torch.isnan(outputs_resized).any()}")

            outputs_sigmoid_resized = torch.sigmoid(outputs_resized)
            print(
                f"Resized sigmoid output range: [{outputs_sigmoid_resized.min()}, {outputs_sigmoid_resized.max()}]"
            )

    break
