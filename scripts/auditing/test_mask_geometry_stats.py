from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path(r"C:\Users\MTC\Downloads\KolektorSDD2\test")


def test_mask_geometry_stats():
    mask_files = sorted(DATASET_DIR.glob("*_GT.png"))

    assert mask_files, "No *_GT.png mask files found."

    areas = []
    widths = []
    heights = []
    aspect_ratios = []
    component_counts = []

    non_empty_masks = 0

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        assert mask is not None, f"Failed to read mask: {mask_path.name}"

        binary = (mask > 0).astype(np.uint8)

        if not np.any(binary):
            continue

        non_empty_masks += 1

        ys, xs = np.where(binary > 0)

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = int(np.sum(binary))

        aspect_ratio = width / height

        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        component_count = num_labels - 1

        areas.append(area)
        widths.append(width)
        heights.append(height)
        aspect_ratios.append(aspect_ratio)
        component_counts.append(component_count)

    assert non_empty_masks > 0, "No non-empty masks found."

    print("\n=== MASK GEOMETRY STATISTICS ===")

    print(f"Total masks audited : {len(mask_files)}")
    print(f"Non-empty masks     : {non_empty_masks}")

    print("\nDefect Area (pixels):")
    print(f"  Min               : {min(areas)}")
    print(f"  Max               : {max(areas)}")
    print(f"  Mean              : {np.mean(areas):.2f}")
    print(f"  Median            : {np.median(areas):.2f}")

    print("\nBounding Box Width:")
    print(f"  Min               : {min(widths)}")
    print(f"  Max               : {max(widths)}")
    print(f"  Mean              : {np.mean(widths):.2f}")
    print(f"  Median            : {np.median(widths):.2f}")

    print("\nBounding Box Height:")
    print(f"  Min               : {min(heights)}")
    print(f"  Max               : {max(heights)}")
    print(f"  Mean              : {np.mean(heights):.2f}")
    print(f"  Median            : {np.median(heights):.2f}")

    print("\nAspect Ratio (width / height):")
    print(f"  Min               : {min(aspect_ratios):.4f}")
    print(f"  Max               : {max(aspect_ratios):.4f}")
    print(f"  Mean              : {np.mean(aspect_ratios):.4f}")
    print(f"  Median            : {np.median(aspect_ratios):.4f}")

    print("\nConnected Components:")
    print(f"  Min               : {min(component_counts)}")
    print(f"  Max               : {max(component_counts)}")
    print(f"  Mean              : {np.mean(component_counts):.2f}")
    print(f"  Median            : {np.median(component_counts):.2f}")