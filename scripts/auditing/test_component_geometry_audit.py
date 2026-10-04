from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path(r"C:\Users\MTC\Downloads\KolektorSDD2\test")


def test_component_geometry_audit():
    mask_files = sorted(DATASET_DIR.glob("*_GT.png"))

    assert mask_files, "No *_GT.png mask files found."

    component_areas = []
    component_widths = []
    component_heights = []

    total_components = 0
    multi_component_masks = 0

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        assert mask is not None, f"Failed to read mask: {mask_path.name}"

        binary = (mask > 0).astype(np.uint8)

        if not np.any(binary):
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        component_count = num_labels - 1

        if component_count > 1:
            multi_component_masks += 1

        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            component_areas.append(int(area))
            component_widths.append(int(width))
            component_heights.append(int(height))

            total_components += 1

    assert total_components > 0, "No connected components found."

    print("\n=== COMPONENT GEOMETRY AUDIT ===")

    print(f"Total components       : {total_components}")
    print(f"Multi-component masks  : {multi_component_masks}")

    print("\nComponent Area:")
    print(f"  Min                  : {min(component_areas)}")
    print(f"  Max                  : {max(component_areas)}")
    print(f"  Mean                 : {np.mean(component_areas):.2f}")
    print(f"  Median               : {np.median(component_areas):.2f}")

    print("\nComponent Width:")
    print(f"  Min                  : {min(component_widths)}")
    print(f"  Max                  : {max(component_widths)}")
    print(f"  Mean                 : {np.mean(component_widths):.2f}")
    print(f"  Median               : {np.median(component_widths):.2f}")

    print("\nComponent Height:")
    print(f"  Min                  : {min(component_heights)}")
    print(f"  Max                  : {max(component_heights)}")
    print(f"  Mean                 : {np.mean(component_heights):.2f}")
    print(f"  Median               : {np.median(component_heights):.2f}")