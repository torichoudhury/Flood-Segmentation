import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

# Define dataset paths
base_path = r"D:\Flood Segmentation\v1.1\data\flood_events\HandLabeled"

# Function to load TIFF images as NumPy arrays
def load_tif_image(filepath):
    with rasterio.open(filepath) as img:
        return img.read(1)  # Read first band (grayscale)

# Function to normalize images
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Function to resize images to a fixed size
def resize_image(image, target_size=(256, 256)):
    return resize(image, target_size, anti_aliasing=True)

# Custom dataset class
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
        mask = resize_image(mask)  # Do not normalize mask

        # Convert to tensor (without ToTensor)
        sar_tensor = torch.tensor(sar_image, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]
        optical_tensor = torch.tensor(optical_image, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0)  # Ensure same shape [1, 256, 256]

        return sar_tensor, optical_tensor, mask_tensor

# Define dataset paths
sar_dir = os.path.join(base_path, "S1Hand")
optical_dir = os.path.join(base_path, "S2Hand")
mask_dir = os.path.join(base_path, "LabelHand")

# Create dataset
dataset = FloodDataset(sar_dir, optical_dir, mask_dir)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Fetch a batch
for sar_batch, optical_batch, mask_batch in dataloader:
    print("✅ SAR Batch Shape:", sar_batch.shape)  # Expected: [8, 1, 256, 256]
    print("✅ Optical Batch Shape:", optical_batch.shape)  # Expected: [8, 1, 256, 256]
    print("✅ Mask Batch Shape:", mask_batch.shape)  # Expected: [8, 1, 256, 256]
    break
