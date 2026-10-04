import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"

CROP_SIZES = [128, 192, 256]

rows = []


def analyze_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "defect_pixels": len(xs),
    }


def crop_around_bbox(mask, crop_size):
    info = analyze_mask(mask)

    if info is None:
        return None

    h, w = mask.shape

    cx = (info["x_min"] + info["x_max"]) // 2
    cy = (info["y_min"] + info["y_max"]) // 2

    half = crop_size // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    # Shift crop back inside image boundaries.
    if x0 < 0:
        x1 -= x0
        x0 = 0

    if y0 < 0:
        y1 -= y0
        y0 = 0

    if x1 > w:
        x0 -= x1 - w
        x1 = w

    if y1 > h:
        y0 -= y1 - h
        y1 = h

    x0 = max(0, x0)
    y0 = max(0, y0)

    cropped = mask[y0:y1, x0:x1]

    return cropped


def main():
    with open(TRAIN_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    defective_rows = [
        row for row in rows
        if row["has_defect"].strip() == "1"
    ]

    print("POSITIVE-AWARE CROPPING AUDIT")
    print("==============================")
    print(f"Train samples:       {len(rows)}")
    print(f"Defective samples:   {len(defective_rows)}")
    print()

    original_ratios = []

    for row in defective_rows:
        mask_path = ROOT / row["mask_path"]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        info = analyze_mask(mask)

        if info is None:
            raise RuntimeError(
                f"Manifest says defective but mask is empty: {mask_path}"
            )

        h, w = mask.shape

        original_ratios.append(
            info["defect_pixels"] / (h * w)
        )

    print("ORIGINAL DEFECT AREA RATIO")
    print("--------------------------")
    print(f"Mean:      {np.mean(original_ratios):.6f}")
    print(f"Median:    {np.median(original_ratios):.6f}")
    print(f"P25:       {np.percentile(original_ratios, 25):.6f}")
    print(f"P75:       {np.percentile(original_ratios, 75):.6f}")
    print()

    for crop_size in CROP_SIZES:
        ratios = []
        bbox_coverage = []
        valid_count = 0

        for row in defective_rows:
            mask_path = ROOT / row["mask_path"]

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            cropped = crop_around_bbox(mask, crop_size)

            if cropped is None:
                continue

            valid_count += 1

            defect_pixels = np.count_nonzero(cropped)

            ratio = defect_pixels / cropped.size

            ratios.append(ratio)

            info = analyze_mask(cropped)

            if info is not None:
                coverage = (
                    info["bbox_width"] * info["bbox_height"]
                ) / cropped.size

                bbox_coverage.append(coverage)

        print(f"CROP SIZE: {crop_size}x{crop_size}")
        print("--------------------------")
        print(f"Valid samples: {valid_count}")
        print(
            f"Defect ratio mean:   {np.mean(ratios):.6f}"
        )
        print(
            f"Defect ratio median: {np.median(ratios):.6f}"
        )
        print(
            f"Defect ratio P25:    {np.percentile(ratios, 25):.6f}"
        )
        print(
            f"Defect ratio P75:    {np.percentile(ratios, 75):.6f}"
        )
        print(
            f"BBox coverage mean:  {np.mean(bbox_coverage):.6f}"
        )
        print(
            f"BBox coverage median:{np.median(bbox_coverage):.6f}"
        )
        print()

    print("AUDIT COMPLETED")


if __name__ == "__main__":
    main()