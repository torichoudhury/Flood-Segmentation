import os
import glob
import numpy as np
import rasterio
from skimage.transform import resize
from dataset import normalize_image  # Import the updated function

# ✅ Define dataset paths
BASE_PATH = r"D:\Flood-Segmentation\dataset\HandLabeled"
SAR_DIR = os.path.join(BASE_PATH, "S1Hand")  # SAR Images


def load_tif_image(filepath):
    with rasterio.open(filepath) as img:
        return img.read(1)  # Read first band (grayscale)


def resize_image(image, target_size=(256, 256)):
    return resize(image, target_size, anti_aliasing=True)


# Test loading the first SAR image
sar_images = sorted(glob.glob(os.path.join(SAR_DIR, "*.tif")))
print(f"Found {len(sar_images)} SAR images")

if len(sar_images) > 0:
    first_sar = sar_images[0]
    print(f"Loading: {first_sar}")

    # Load raw image
    raw_image = load_tif_image(first_sar)
    print(f"Raw image shape: {raw_image.shape}")
    print(f"Raw image dtype: {raw_image.dtype}")
    print(f"Raw image range: [{np.min(raw_image)}, {np.max(raw_image)}]")
    print(f"Raw image has NaN: {np.isnan(raw_image).any()}")
    print(f"Raw image has Inf: {np.isinf(raw_image).any()}")

    # Resize first
    resized_image = resize_image(raw_image)
    print(f"Resized image shape: {resized_image.shape}")
    print(f"Resized image range: [{np.min(resized_image)}, {np.max(resized_image)}]")
    print(f"Resized image has NaN: {np.isnan(resized_image).any()}")

    # Normalize
    normalized_image = normalize_image(resized_image)
    print(
        f"Normalized image range: [{np.min(normalized_image)}, {np.max(normalized_image)}]"
    )
    print(f"Normalized image has NaN: {np.isnan(normalized_image).any()}")
else:
    print("No SAR images found!")
