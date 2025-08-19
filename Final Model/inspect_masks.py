import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from glob import glob


def inspect_mask_files():
    """Inspect the actual mask files to understand the data."""

    mask_dir = r"D:\Flood-Segmentation\dataset\HandLabeled\LabelHand"
    mask_files = sorted(glob(os.path.join(mask_dir, "*.tif")))

    print(f"Found {len(mask_files)} mask files")
    print("\nInspecting first 10 mask files:")
    print("=" * 60)

    for i, mask_file in enumerate(mask_files[:10]):
        filename = os.path.basename(mask_file)

        try:
            with rasterio.open(mask_file) as img:
                mask = img.read(1)  # Read first band

            print(f"\nFile {i+1}: {filename}")
            print(f"  Shape: {mask.shape}")
            print(f"  Data type: {mask.dtype}")
            print(f"  Value range: [{np.min(mask):.6f}, {np.max(mask):.6f}]")
            print(f"  Unique values: {np.unique(mask)}")
            print(f"  Non-zero pixels: {np.count_nonzero(mask):,}")
            print(f"  Total pixels: {mask.size:,}")

            if np.max(mask) > 0:
                print(f"  ✅ HAS FLOOD PIXELS!")

                # Save visualization of first mask with flood pixels
                if i == 0 or np.max(mask) > 0:
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(mask, cmap="Blues")
                    plt.title(f"Raw Mask: {filename}")
                    plt.colorbar()

                    plt.subplot(1, 2, 2)
                    binary_mask = (mask > 0).astype(float)
                    plt.imshow(binary_mask, cmap="Blues")
                    plt.title(f"Binary Mask (>0)")
                    plt.colorbar()

                    plt.tight_layout()
                    plt.savefig(
                        f"mask_inspection_{i+1}.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close()
                    print(f"  📷 Saved visualization: mask_inspection_{i+1}.png")
            else:
                print(f"  ❌ NO FLOOD PIXELS")

        except Exception as e:
            print(f"  ❌ ERROR reading file: {e}")

    # Check if any files have flood pixels
    has_flood_files = []
    print(f"\n🔍 SCANNING ALL {len(mask_files)} FILES FOR FLOOD PIXELS...")

    for i, mask_file in enumerate(mask_files):
        try:
            with rasterio.open(mask_file) as img:
                mask = img.read(1)
            if np.max(mask) > 0:
                has_flood_files.append(
                    (mask_file, np.max(mask), np.count_nonzero(mask))
                )
        except:
            pass

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(mask_files)} files...")

    print(f"\n📊 SUMMARY:")
    print(f"Files with flood pixels: {len(has_flood_files)}")

    if len(has_flood_files) == 0:
        print("❌ CRITICAL: NO mask files contain flood pixels!")
        print(
            "   This explains why your model can't learn - there are no positive examples."
        )
        print("\n🔧 POSSIBLE SOLUTIONS:")
        print("   1. Check if you're using the correct mask directory")
        print("   2. Verify mask files aren't corrupted")
        print("   3. Check if flood pixels use different values (not 1)")
        print("   4. Consider using JRCWaterHand files as ground truth")
    else:
        print("✅ Found files with flood pixels:")
        for file, max_val, count in has_flood_files[:5]:  # Show first 5
            filename = os.path.basename(file)
            print(f"   {filename}: max={max_val:.6f}, flood_pixels={count:,}")
        if len(has_flood_files) > 5:
            print(f"   ... and {len(has_flood_files) - 5} more files")


if __name__ == "__main__":
    inspect_mask_files()
