from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "splits" / "train.csv",
    "validation": ROOT / "splits" / "validation.csv",
}

TARGET_SIZES = [256, 320, 384]


def get_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def audit_split(csv_path, split_name):
    df = pd.read_csv(csv_path)

    records = []

    for _, row in df.iterrows():
        if int(row["has_defect"]) == 0:
            continue

        mask = cv2.imread(
            str(row["mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Failed to read mask: {row['mask_path']}"
            )

        bbox = get_bbox(mask)

        if bbox is None:
            raise RuntimeError(
                f"Defective sample has empty mask: "
                f"{row['mask_path']}"
            )

        x_min, y_min, x_max, y_max = bbox

        height, width = mask.shape

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        area = int(np.count_nonzero(mask))

        record = {
            "image_id": row["image_id"],
            "width": width,
            "height": height,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "area": area,
        }

        for target in TARGET_SIZES:
            scale = min(
                target / width,
                target / height,
            )

            record[f"{target}_bbox_width"] = bbox_width * scale
            record[f"{target}_bbox_height"] = bbox_height * scale
            record[f"{target}_area"] = area * scale * scale

        records.append(record)

    result = pd.DataFrame(records)

    print(f"=== {split_name.upper()} DEFECT RESOLUTION AUDIT ===")
    print(f"Defective samples    : {len(result)}")
    print()

    print("Original defect bbox:")
    print(
        f"  Width  median     : "
        f"{result['bbox_width'].median():.2f}"
    )
    print(
        f"  Height median     : "
        f"{result['bbox_height'].median():.2f}"
    )

    print()

    for target in TARGET_SIZES:
        width_col = f"{target}_bbox_width"
        height_col = f"{target}_bbox_height"
        area_col = f"{target}_area"

        print(f"--- Target {target}x{target} ---")

        print(
            f"  BBox width  P10/P25/Median : "
            f"{result[width_col].quantile(0.10):.2f} / "
            f"{result[width_col].quantile(0.25):.2f} / "
            f"{result[width_col].median():.2f}"
        )

        print(
            f"  BBox height P10/P25/Median : "
            f"{result[height_col].quantile(0.10):.2f} / "
            f"{result[height_col].quantile(0.25):.2f} / "
            f"{result[height_col].median():.2f}"
        )

        print(
            f"  Area P10/P25/Median         : "
            f"{result[area_col].quantile(0.10):.2f} / "
            f"{result[area_col].quantile(0.25):.2f} / "
            f"{result[area_col].median():.2f}"
        )

        print()


if __name__ == "__main__":
    for split_name, csv_path in SPLITS.items():
        audit_split(csv_path, split_name)
        print()

    print("DEFECT RESOLUTION AUDIT COMPLETED")