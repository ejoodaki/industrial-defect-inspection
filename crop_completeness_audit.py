import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "splits" / "train.csv"

CROP_SIZES = [128, 192, 256]


def get_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        "pixels": int(len(xs)),
    }


def make_crop(mask, crop_size):
    bbox = get_bbox(mask)

    if bbox is None:
        return None, None

    h, w = mask.shape

    cx = (bbox["x_min"] + bbox["x_max"]) // 2
    cy = (bbox["y_min"] + bbox["y_max"]) // 2

    half = crop_size // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

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

    x1 = min(w, x1)
    y1 = min(h, y1)

    crop = mask[y0:y1, x0:x1]

    return crop, (x0, y0, x1, y1)


def main():
    with open(TRAIN_CSV, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    defective_rows = [
        row for row in rows
        if row["has_defect"].strip() == "1"
    ]

    print("CROP COMPLETENESS AUDIT")
    print("========================")
    print(f"Defective samples: {len(defective_rows)}")
    print()

    for crop_size in CROP_SIZES:

        complete_count = 0
        clipped_count = 0
        total_original_pixels = 0
        total_kept_pixels = 0

        bbox_touch_count = 0

        completeness_values = []

        for row in defective_rows:

            mask_path = ROOT / row["mask_path"]

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is None:
                raise RuntimeError(
                    f"Failed to read mask: {mask_path}"
                )

            bbox = get_bbox(mask)

            if bbox is None:
                continue

            original_pixels = bbox["pixels"]

            crop, coords = make_crop(
                mask,
                crop_size,
            )

            if crop is None:
                continue

            x0, y0, x1, y1 = coords

            kept_pixels = int(np.count_nonzero(crop))

            completeness = (
                kept_pixels / original_pixels
            )

            completeness_values.append(completeness)

            total_original_pixels += original_pixels
            total_kept_pixels += kept_pixels

            if completeness >= 0.999999:
                complete_count += 1
            else:
                clipped_count += 1

            crop_h, crop_w = crop.shape

            crop_bbox = get_bbox(crop)

            if crop_bbox is not None:
                touches_edge = (
                    crop_bbox["x_min"] == 0
                    or crop_bbox["y_min"] == 0
                    or crop_bbox["x_max"] == crop_w - 1
                    or crop_bbox["y_max"] == crop_h - 1
                )

                if touches_edge:
                    bbox_touch_count += 1

        total = complete_count + clipped_count

        print(f"CROP SIZE: {crop_size}x{crop_size}")
        print("--------------------------")

        print(f"Complete defect:      {complete_count}/{total}")
        print(
            f"Clipped defect:       {clipped_count}/{total}"
        )

        print(
            f"Completeness rate:    "
            f"{complete_count / total:.4f}"
        )

        print(
            f"Mean completeness:    "
            f"{np.mean(completeness_values):.4f}"
        )

        print(
            f"Median completeness:  "
            f"{np.median(completeness_values):.4f}"
        )

        print(
            f"P10 completeness:     "
            f"{np.percentile(completeness_values, 10):.4f}"
        )

        print(
            f"P25 completeness:     "
            f"{np.percentile(completeness_values, 25):.4f}"
        )

        print(
            f"BBox touches edge:    "
            f"{bbox_touch_count}/{total}"
        )

        print(
            f"Pixels retained:      "
            f"{total_kept_pixels / total_original_pixels:.4f}"
        )

        print()

    print("AUDIT COMPLETED")


if __name__ == "__main__":
    main()