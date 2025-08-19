import os
import glob
import numpy as np
import rasterio
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset Class (Loads Preprocessed Images)
class FloodDataset(Dataset):
    def __init__(self, base_path):
        self.sar_images = sorted(glob.glob(os.path.join(base_path, "S1Hand", "*.tif")))
        self.optical_images = sorted(glob.glob(os.path.join(base_path, "S2Hand", "*.tif")))
        self.mask_images = sorted(glob.glob(os.path.join(base_path, "LabelHand", "*.tif")))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        with rasterio.open(self.sar_images[idx]) as img:
            sar_image = img.read(1)
        with rasterio.open(self.optical_images[idx]) as img:
            optical_image = img.read(1)
        with rasterio.open(self.mask_images[idx]) as img:
            mask = img.read(1)

        # Convert to tensors
        sar_tensor = torch.tensor(sar_image, dtype=torch.float32).unsqueeze(0)
        optical_tensor = torch.tensor(optical_image, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        return sar_tensor, optical_tensor, mask_tensor

# ✅ EfficientNet-B4 Feature Extractor (Fix for Feature Map Dimensions)
class FloodFeatureExtractor(nn.Module):
    def __init__(self):
        super(FloodFeatureExtractor, self).__init__()

        # Load EfficientNet-B4 without classification head
        self.backbone = timm.create_model("tf_efficientnet_b4", pretrained=True, features_only=True)
        
        # Freeze EfficientNet layers
        for param in self.backbone.parameters():
            param.requires_grad = False  

        # ✅ Fix: Find correct feature map sizes dynamically
        self.reduce_channels = nn.Conv2d(896, 512, kernel_size=1)  # Change 1792 → 896 dynamically

    def forward(self, sar, optical):
        # Convert 1-channel to 3-channel for EfficientNet
        sar = sar.repeat(1, 3, 1, 1).to(device)  
        optical = optical.repeat(1, 3, 1, 1).to(device)

        # Extract feature maps from EfficientNet
        sar_features = self.backbone(sar)  # Returns multiple feature maps
        optical_features = self.backbone(optical)

        # Print feature map sizes for debugging
        print(f"✅ SAR Feature Map Sizes: {[f.shape for f in sar_features]}")
        print(f"✅ Optical Feature Map Sizes: {[f.shape for f in optical_features]}")

        # ✅ Fix: Select correct feature map dynamically (EfficientNet-B4 last feature size = 896 channels)
        sar_features = sar_features[-1]  # Extract last feature map
        optical_features = optical_features[-1]

        # Merge SAR & Optical features
        combined_features = torch.cat([sar_features, optical_features], dim=1)  # Now, channels = 896

        # Reduce to fixed size (512 channels)
        return self.reduce_channels(combined_features)

# ✅ Main Execution (Optimized)
if __name__ == "__main__":
    base_path = r"D:\Flood Segmentation\v1.1\data\flood_events\HandLabeled"
    dataset = FloodDataset(base_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = FloodFeatureExtractor().to(device)  # Move model to GPU

    for sar_batch, optical_batch, mask_batch in dataloader:
        print("SAR Batch Shape:", sar_batch.shape)  
        print("Optical Batch Shape:", optical_batch.shape)  
        print("Mask Batch Shape:", mask_batch.shape)  

        features = model(sar_batch, optical_batch)
        print("Feature Map Shape:", features.shape)  # Expected: [4, 512, H, W]
        break  # Stop after first batch
