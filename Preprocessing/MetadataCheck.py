import rasterio

def print_metadata(file_path):
    with rasterio.open(file_path) as src:
        print(f"\n📂 File: {file_path}")
        print(f"🗺️ CRS: {src.crs}")  # Coordinate Reference System
        print(f"📍 Bounds: {src.bounds}")  # Geospatial extent
        print(f"📏 Resolution: {src.res}")  # Pixel size (meters per pixel)
        print(f"📐 Shape: {src.shape}")  # (Height, Width)

# ✅ Update paths based on your dataset structure
base_path = r"D:\Flood Segmentation\v1.1\data\flood_events\HandLabeled"

sar_path = f"{base_path}\S1Hand\India_1017769_S1Hand.tif"   # SAR Image
optical_path = f"{base_path}\S2Hand\India_1017769_S2Hand.tif"   # Optical Image
mask_path = f"{base_path}\LabelHand\India_1017769_LabelHand.tif"  # Flood Mask

# 🛠 Print metadata for each image
print_metadata(sar_path)
print_metadata(optical_path)
print_metadata(mask_path)



"""
 Conclusion
✔ Good News: The SAR, Optical, and Mask images have the same shape (512, 512).
✔ Same CRS (EPSG:4326) → No need for re-projection.
✔ Same Bounding Box (Spatial Extent) → No need for cropping or shifting.

🔹 Minor Issue:

The resolution of the Flood Mask is slightly different (8.983152841195857e-05) vs. (8.983152841195215e-05) for SAR/Optical.
This difference is extremely small and likely won't affect training.
"""