import torch
import torch.nn.functional as F
from model import FloodSegmentationModel
from dataset import FloodDataset
from loss import AdaptiveLoss, IoU
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

# Model and Loss Initialization
model = FloodSegmentationModel().to(device)
criterion = AdaptiveLoss().to(device)
iou_metric = IoU().to(device)

# Test one batch
for sar_images, optical_images, masks in train_loader:
    sar_images, optical_images, masks = (
        sar_images.to(device),
        optical_images.to(device),
        masks.to(device),
    )

    print(
        f"Input shapes - SAR: {sar_images.shape}, Optical: {optical_images.shape}, Mask: {masks.shape}"
    )

    # Get model outputs
    outputs = model(sar_images, optical_images)
    print(f"Model output shape (before resize): {outputs.shape}")

    # Resize outputs to match mask size if needed
    if outputs.shape[-2:] != masks.shape[-2:]:
        print("Resizing outputs to match mask size...")
        outputs = F.interpolate(
            outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        print(f"Model output shape (after resize): {outputs.shape}")

    # Test loss function
    try:
        loss, loss_dict = criterion(outputs, masks, 0)  # epoch=0
        iou = iou_metric(outputs, masks)
        print(f"Loss: {loss.item()}, IoU: {iou.item()}")
        print("SUCCESS: Loss and IoU calculated correctly!")
    except Exception as e:
        print(f"ERROR in loss calculation: {e}")

        # Debug tensor details
        print(f"Outputs dtype: {outputs.dtype}, shape: {outputs.shape}")
        print(f"Masks dtype: {masks.dtype}, shape: {masks.shape}")
        print(f"Outputs range: [{outputs.min()}, {outputs.max()}]")
        print(f"Masks range: [{masks.min()}, {masks.max()}]")
        print(f"Unique values in masks: {torch.unique(masks)}")

    break
