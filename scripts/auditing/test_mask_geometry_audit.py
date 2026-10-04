from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path(r"C:\Users\MTC\Downloads\KolektorSDD2\test")


def test_mask_geometry_audit():
    mask_files = sorted(DATASET_DIR.glob("*_GT.png"))

    assert mask_files, "No *_GT.png mask files found."

    audited = 0
    empty_masks = 0
    invalid_masks = 0

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        assert mask is not None, f"Failed to read mask: {mask_path.name}"

        audited += 1

        unique_values = np.unique(mask)

        if len(unique_values) == 1 and unique_values[0] == 0:
            empty_masks += 1
            continue

        if np.any((unique_values != 0) & (unique_values != 255)):
            invalid_masks += 1

    print("\n=== MASK GEOMETRY AUDIT ===")
    print(f"Total masks audited : {audited}")
    print(f"Empty masks         : {empty_masks}")
    print(f"Invalid masks       : {invalid_masks}")

    assert invalid_masks == 0, (
        f"Found {invalid_masks} masks containing values other than 0/255."
    )