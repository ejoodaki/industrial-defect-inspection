from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path(".")


def inspect_mask(mask_path: Path) -> dict:
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        return {
            "valid": False,
            "shape": None,
            "unique_values": [],
            "nonzero_pixels": 0,
        }

    unique_values = np.unique(mask)

    return {
        "valid": True,
        "shape": mask.shape,
        "unique_values": unique_values.tolist(),
        "nonzero_pixels": int(np.count_nonzero(mask)),
    }


def main() -> None:
    image_paths = sorted(
        path
        for path in DATASET_DIR.glob("*.png")
        if not path.name.endswith("_GT.png")
    )

    mask_paths = sorted(
        DATASET_DIR.glob("*_GT.png")
    )

    image_names = {
        path.stem
        for path in image_paths
    }

    mask_names = {
        path.stem.removesuffix("_GT")
        for path in mask_paths
    }

    missing_masks = sorted(
        image_names - mask_names
    )

    orphan_masks = sorted(
        mask_names - image_names
    )

    positive = 0
    negative = 0
    invalid_masks = 0

    image_shapes = set()
    mask_shapes = set()
    defect_pixel_counts = []

    non_binary_masks = []

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is not None:
            image_shapes.add(image.shape)

    for mask_path in mask_paths:
        result = inspect_mask(mask_path)

        if not result["valid"]:
            invalid_masks += 1
            continue

        mask_shapes.add(result["shape"])

        if result["nonzero_pixels"] > 0:
            positive += 1
            defect_pixel_counts.append(
                result["nonzero_pixels"]
            )
        else:
            negative += 1

        if not set(result["unique_values"]).issubset(
            {0, 255}
        ):
            non_binary_masks.append(
                mask_path.name
            )

    print("=" * 60)
    print("KolektorSDD2 Dataset Audit")
    print("=" * 60)

    print()
    print("Dataset directory:")
    print(DATASET_DIR.resolve())

    print()
    print("Files")
    print("-" * 60)
    print(f"Images:       {len(image_paths)}")
    print(f"GT masks:     {len(mask_paths)}")

    print()
    print("Labels")
    print("-" * 60)
    print(f"Positive:     {positive}")
    print(f"Negative:     {negative}")
    print(f"Invalid:      {invalid_masks}")

    print()
    print("Integrity")
    print("-" * 60)
    print(f"Missing GT:   {len(missing_masks)}")
    print(f"Orphan GT:    {len(orphan_masks)}")

    print()
    print("Image shapes")
    print("-" * 60)

    for shape in sorted(image_shapes):
        print(shape)

    print()
    print("Mask shapes")
    print("-" * 60)

    for shape in sorted(mask_shapes):
        print(shape)

    print()
    print("Mask validation")
    print("-" * 60)
    print(
        f"Non-binary masks: {len(non_binary_masks)}"
    )

    if defect_pixel_counts:
        print()
        print("Defect pixel statistics")
        print("-" * 60)
        print(
            f"Min: {min(defect_pixel_counts)}"
        )
        print(
            f"Max: {max(defect_pixel_counts)}"
        )
        print(
            f"Mean: {np.mean(defect_pixel_counts):.2f}"
        )
        print(
            f"Median: {np.median(defect_pixel_counts):.2f}"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()