from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "splits" / "train.csv",
    "validation": ROOT / "splits" / "validation.csv",
}

OUTPUT_DIR = ROOT / "outputs" / "dataset_audit"


def audit_split(csv_path: Path, split_name: str) -> None:
    df = pd.read_csv(csv_path)

    shape_mismatches = []
    invalid_masks = []
    invalid_images = []

    defect_pixel_counts = []

    image_dtypes = set()
    mask_dtypes = set()

    print(f"=== {split_name.upper()} PAIR AUDIT ===")
    print(f"Samples: {len(df)}")
    print()

    for _, row in df.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            invalid_images.append(str(image_path))
            continue

        if mask is None:
            invalid_masks.append(str(mask_path))
            continue

        image_dtypes.add(str(image.dtype))
        mask_dtypes.add(str(mask.dtype))

        # Image/mask geometry must match exactly.
        if image.shape != mask.shape:
            shape_mismatches.append(
                {
                    "image_id": row["image_id"],
                    "image_shape": image.shape,
                    "mask_shape": mask.shape,
                }
            )

        # Mask must contain only 0 and 255.
        unique_values = set(np.unique(mask).tolist())

        if not unique_values.issubset({0, 255}):
            invalid_masks.append(
                f"{mask_path} -> values={sorted(unique_values)}"
            )

        defect_pixels = int(np.count_nonzero(mask == 255))
        defect_pixel_counts.append(defect_pixels)

    print(f"Image dtypes         : {sorted(image_dtypes)}")
    print(f"Mask dtypes          : {sorted(mask_dtypes)}")
    print(f"Shape mismatches     : {len(shape_mismatches)}")
    print(f"Invalid images       : {len(invalid_images)}")
    print(f"Invalid masks        : {len(invalid_masks)}")

    if defect_pixel_counts:
        print(
            f"Defect pixel range   : "
            f"{min(defect_pixel_counts)} - {max(defect_pixel_counts)}"
        )
        print(
            f"Mean defect pixels   : "
            f"{np.mean(defect_pixel_counts):.2f}"
        )

    print()

    if shape_mismatches:
        print("FIRST SHAPE MISMATCH:")
        print(shape_mismatches[0])
        raise RuntimeError(
            f"{split_name}: image/mask shape mismatch detected."
        )

    if invalid_images:
        raise RuntimeError(
            f"{split_name}: invalid images detected."
        )

    if invalid_masks:
        print("FIRST INVALID MASK:")
        print(invalid_masks[0])
        raise RuntimeError(
            f"{split_name}: invalid masks detected."
        )

    print("PASSED")


def save_visual_audit() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLITS["validation"])

    # Select one healthy and one defective sample.
    healthy = df[df["has_defect"] == 0].iloc[0]
    defective = df[df["has_defect"] == 1].iloc[0]

    samples = [
        ("validation_healthy", healthy),
        ("validation_defective", defective),
    ]

    for name, row in samples:
        image = cv2.imread(
            str(row["image_path"]),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            str(row["mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        )

        # Convert grayscale image to BGR for overlay.
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create red overlay where defect mask exists.
        overlay = image_bgr.copy()
        overlay[mask == 255] = (0, 0, 255)

        blended = cv2.addWeighted(
            image_bgr,
            0.7,
            overlay,
            0.3,
            0,
        )

        image_output = OUTPUT_DIR / f"{name}_image.png"
        mask_output = OUTPUT_DIR / f"{name}_mask.png"
        overlay_output = OUTPUT_DIR / f"{name}_overlay.png"

        cv2.imwrite(str(image_output), image)
        cv2.imwrite(str(mask_output), mask)
        cv2.imwrite(str(overlay_output), blended)

        print(f"Saved: {image_output}")
        print(f"Saved: {mask_output}")
        print(f"Saved: {overlay_output}")


if __name__ == "__main__":
    audit_split(SPLITS["train"], "train")
    print()

    audit_split(SPLITS["validation"], "validation")
    print()

    save_visual_audit()

    print()
    print("PAIR AUDIT COMPLETED SUCCESSFULLY")