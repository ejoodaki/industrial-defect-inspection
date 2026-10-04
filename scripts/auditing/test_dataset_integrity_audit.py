from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path(r"C:\Users\MTC\Downloads\KolektorSDD2\test")


def test_dataset_integrity_audit():
    image_files = sorted(
        path
        for path in DATASET_DIR.glob("*.png")
        if not path.name.endswith("_GT.png")
    )

    mask_files = sorted(DATASET_DIR.glob("*_GT.png"))

    assert image_files, "No image files found."
    assert mask_files, "No GT mask files found."

    image_map = {
        path.stem: path
        for path in image_files
    }

    mask_map = {
        path.stem.removesuffix("_GT"): path
        for path in mask_files
    }

    image_without_mask = sorted(set(image_map) - set(mask_map))
    mask_without_image = sorted(set(mask_map) - set(image_map))

    unreadable_images = []
    unreadable_masks = []
    dimension_mismatches = []
    invalid_masks = []

    for image_id in sorted(set(image_map) & set(mask_map)):
        image_path = image_map[image_id]
        mask_path = mask_map[image_id]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            unreadable_images.append(image_path.name)
            continue

        if mask is None:
            unreadable_masks.append(mask_path.name)
            continue

        if image.shape[:2] != mask.shape[:2]:
            dimension_mismatches.append(
                (
                    image_path.name,
                    image.shape[:2],
                    mask.shape[:2],
                )
            )

        unique_values = np.unique(mask)

        if not np.all(np.isin(unique_values, [0, 255])):
            invalid_masks.append(
                (
                    mask_path.name,
                    unique_values.tolist(),
                )
            )

    print("\n=== DATASET INTEGRITY AUDIT ===")

    print(f"Images found          : {len(image_files)}")
    print(f"GT masks found        : {len(mask_files)}")
    print(f"Matched pairs         : {len(set(image_map) & set(mask_map))}")

    print(f"Images without mask   : {len(image_without_mask)}")
    print(f"Masks without image   : {len(mask_without_image)}")
    print(f"Unreadable images     : {len(unreadable_images)}")
    print(f"Unreadable masks      : {len(unreadable_masks)}")
    print(f"Dimension mismatches  : {len(dimension_mismatches)}")
    print(f"Invalid masks         : {len(invalid_masks)}")

    if image_without_mask:
        print("\nImages without masks:")
        for name in image_without_mask[:10]:
            print(f"  {name}")

    if mask_without_image:
        print("\nMasks without images:")
        for name in mask_without_image[:10]:
            print(f"  {name}")

    if dimension_mismatches:
        print("\nDimension mismatches:")
        for item in dimension_mismatches[:10]:
            print(f"  {item}")

    if invalid_masks:
        print("\nInvalid masks:")
        for item in invalid_masks[:10]:
            print(f"  {item}")

    assert not image_without_mask, (
        f"Found {len(image_without_mask)} images without matching GT masks."
    )

    assert not mask_without_image, (
        f"Found {len(mask_without_image)} GT masks without matching images."
    )

    assert not unreadable_images, (
        f"Found {len(unreadable_images)} unreadable images."
    )

    assert not unreadable_masks, (
        f"Found {len(unreadable_masks)} unreadable masks."
    )

    assert not dimension_mismatches, (
        f"Found {len(dimension_mismatches)} image/mask dimension mismatches."
    )

    assert not invalid_masks, (
        f"Found {len(invalid_masks)} invalid masks."
    )