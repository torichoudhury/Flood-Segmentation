import os
import glob
import numpy as np
import rasterio
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

# ✅ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset paths
BASE_PATH = r"D:\Flood-Segmentation\dataset\HandLabeled"
SAR_DIR = os.path.join(BASE_PATH, "S1Hand")  # SAR Images
OPTICAL_DIR = os.path.join(BASE_PATH, "S2Hand")  # Optical Images
MASK_DIR = os.path.join(BASE_PATH, "LabelHand")  # Flood Masks


# ✅ Function to load TIFF images as NumPy arrays
def load_tif_image(filepath):
    with rasterio.open(filepath) as img:
        return img.read(1)  # Read first band (grayscale)


# ✅ Function to normalize images
def normalize_image(image):
    # Handle NaN values by replacing them with 0
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Check if image is constant (all values same)
    if np.max(image) == np.min(image):
        return image * 0.0  # Return zeros if constant

    return (image - np.min(image)) / (np.max(image) - np.min(image))


# ✅ Function to resize images
def resize_image(image, target_size=(256, 256)):
    return resize(image, target_size, anti_aliasing=True)


# ✅ Flood Dataset Class
class FloodDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, mask_dir):
        self.sar_images = sorted(glob.glob(os.path.join(sar_dir, "*.tif")))
        self.optical_images = sorted(glob.glob(os.path.join(optical_dir, "*.tif")))
        self.mask_images = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Load images
        sar_image = load_tif_image(self.sar_images[idx])
        optical_image = load_tif_image(self.optical_images[idx])
        mask = load_tif_image(self.mask_images[idx])

        # Normalize and resize
        sar_image = normalize_image(resize_image(sar_image))
        optical_image = normalize_image(resize_image(optical_image))

        # Process mask correctly: -1, 0, 1 → 0, 0, 1 → resize → threshold properly
        # First, convert -1 (unknown) to 0 (no flood), keep 1 as flood
        mask = np.where(mask == 1, 1, 0).astype(
            np.float32
        )  # 1=flood, everything else=0
        mask = resize_image(mask)  # Resize flood mask
        mask = (mask > 0.3).astype(
            np.float32
        )  # Lower threshold to preserve flood pixels after resize        # Convert to tensor
        sar_tensor = torch.tensor(sar_image, dtype=torch.float32).unsqueeze(0)
        optical_tensor = torch.tensor(optical_image, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(
            0
        )  # Changed to float32

        return sar_tensor, optical_tensor, mask_tensor


# ✅ EfficientNet-B4 Feature Extractor
class FloodFeatureExtractor(nn.Module):
    def __init__(self):
        super(FloodFeatureExtractor, self).__init__()

        # Load EfficientNet-B4 without classification head
        self.backbone = timm.create_model(
            "tf_efficientnet_b4", pretrained=True, features_only=True
        )

        # Freeze EfficientNet layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Reduce channels dynamically
        self.reduce_channels = nn.Conv2d(896, 512, kernel_size=1)

    def forward(self, sar, optical):
        # Convert 1-channel to 3-channel for EfficientNet
        sar = sar.repeat(1, 3, 1, 1).to(device)
        optical = optical.repeat(1, 3, 1, 1).to(device)

        # Extract feature maps from EfficientNet
        sar_features = self.backbone(sar)
        optical_features = self.backbone(optical)

        # Select last feature map dynamically
        sar_features = sar_features[-1]
        optical_features = optical_features[-1]

        # Merge SAR & Optical features
        combined_features = torch.cat([sar_features, optical_features], dim=1)

        # Reduce to fixed size (512 channels)
        return self.reduce_channels(combined_features)


# ✅ Main Execution (Testing the Dataset and Model)
if __name__ == "__main__":
    dataset = FloodDataset(SAR_DIR, OPTICAL_DIR, MASK_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = FloodFeatureExtractor().to(device)

    for sar_batch, optical_batch, mask_batch in dataloader:
        print("SAR Batch Shape:", sar_batch.shape)
        print("Optical Batch Shape:", optical_batch.shape)
        print("Mask Batch Shape:", mask_batch.shape)

        features = model(sar_batch, optical_batch)
        print("Feature Map Shape:", features.shape)  # Expected: [4, 512, H, W]
        break  # Stop after first batch
