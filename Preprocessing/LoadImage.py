import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Define dataset paths (HandLabeled)
DATASET_DIR = "D:/Flood Segmentation/v1.1/data/flood_events/HandLabeled"
SAR_DIR = os.path.join(DATASET_DIR, "S1Hand")  # SAR Images
OPTICAL_DIR = os.path.join(DATASET_DIR, "S2Hand")  # Optical Images
MASK_DIR = os.path.join(DATASET_DIR, "LabelHand")  # Flood Masks

# Function to load TIFF images
def load_tif_image(filepath):
    with rasterio.open(filepath) as img:
        return img.read(1)  # Read first band (grayscale for SAR)

# Get example files (first file in each folder)
sar_example_path = os.path.join(SAR_DIR, os.listdir(SAR_DIR)[0])
optical_example_path = os.path.join(OPTICAL_DIR, os.listdir(OPTICAL_DIR)[0])
mask_example_path = os.path.join(MASK_DIR, os.listdir(MASK_DIR)[0])

# Load images
sar_image = load_tif_image(sar_example_path)
optical_image = load_tif_image(optical_example_path)
mask = load_tif_image(mask_example_path)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(sar_image, cmap="gray")
plt.title("SAR Image (S1Hand)")

plt.subplot(1, 3, 2)
plt.imshow(optical_image)
plt.title("Optical Image (S2Hand)")

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap="gray")
plt.title("Flood Mask (LabelHand)")

plt.show()
