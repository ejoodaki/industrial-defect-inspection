from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "splits" / "train.csv"
VAL_CSV = ROOT / "splits" / "validation.csv"


def audit_split(csv_path: Path, name: str) -> None:
    df = pd.read_csv(csv_path)

    image_heights = []
    image_widths = []
    mask_values = set()
    mask_shapes = set()

    for _, row in df.iterrows():
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to read image: {row['image_path']}")

        if mask is None:
            raise RuntimeError(f"Failed to read mask: {row['mask_path']}")

        image_heights.append(image.shape[0])
        image_widths.append(image.shape[1])
        mask_shapes.add(mask.shape)
        mask_values.update(np.unique(mask).tolist())

    print(f"=== {name.upper()} DATASET AUDIT ===")
    print(f"Samples              : {len(df)}")
    print(
        f"Image height range   : "
        f"{min(image_heights)} - {max(image_heights)}"
    )
    print(
        f"Image width range    : "
        f"{min(image_widths)} - {max(image_widths)}"
    )
    print(f"Unique mask shapes   : {sorted(mask_shapes)}")
    print(f"Unique mask values   : {sorted(mask_values)}")
    print("PASSED")


if __name__ == "__main__":
    audit_split(TRAIN_CSV, "train")
    print()
    audit_split(VAL_CSV, "validation")