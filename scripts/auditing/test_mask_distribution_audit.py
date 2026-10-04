from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path(r"C:\Users\MTC\Downloads\KolektorSDD2\test")


def test_mask_distribution_audit():
    mask_files = sorted(DATASET_DIR.glob("*_GT.png"))

    assert mask_files, "No *_GT.png mask files found."

    areas = []
    widths = []
    heights = []

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        assert mask is not None, f"Failed to read mask: {mask_path.name}"

        binary = (mask > 0).astype(np.uint8)

        if not np.any(binary):
            continue

        ys, xs = np.where(binary > 0)

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        area = int(np.sum(binary))
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        areas.append(area)
        widths.append(width)
        heights.append(height)

    assert areas, "No non-empty masks found."

    area_bins = {
        "<500": 0,
        "500-999": 0,
        "1,000-2,499": 0,
        "2,500-4,999": 0,
        "5,000-9,999": 0,
        ">=10,000": 0,
    }

    width_bins = {
        "<25": 0,
        "25-49": 0,
        "50-99": 0,
        "100-149": 0,
        ">=150": 0,
    }

    height_bins = {
        "<25": 0,
        "25-49": 0,
        "50-99": 0,
        "100-199": 0,
        ">=200": 0,
    }

    for area in areas:
        if area < 500:
            area_bins["<500"] += 1
        elif area < 1000:
            area_bins["500-999"] += 1
        elif area < 2500:
            area_bins["1,000-2,499"] += 1
        elif area < 5000:
            area_bins["2,500-4,999"] += 1
        elif area < 10000:
            area_bins["5,000-9,999"] += 1
        else:
            area_bins[">=10,000"] += 1

    for width in widths:
        if width < 25:
            width_bins["<25"] += 1
        elif width < 50:
            width_bins["25-49"] += 1
        elif width < 100:
            width_bins["50-99"] += 1
        elif width < 150:
            width_bins["100-149"] += 1
        else:
            width_bins[">=150"] += 1

    for height in heights:
        if height < 25:
            height_bins["<25"] += 1
        elif height < 50:
            height_bins["25-49"] += 1
        elif height < 100:
            height_bins["50-99"] += 1
        elif height < 200:
            height_bins["100-199"] += 1
        else:
            height_bins[">=200"] += 1

    print("\n=== MASK DISTRIBUTION AUDIT ===")

    print(f"Non-empty masks: {len(areas)}")

    print("\nDefect Area Distribution:")
    for bucket, count in area_bins.items():
        percentage = (count / len(areas)) * 100
        print(f"  {bucket:>12} : {count:3d} ({percentage:5.1f}%)")

    print("\nBounding Box Width Distribution:")
    for bucket, count in width_bins.items():
        percentage = (count / len(widths)) * 100
        print(f"  {bucket:>12} : {count:3d} ({percentage:5.1f}%)")

    print("\nBounding Box Height Distribution:")
    for bucket, count in height_bins.items():
        percentage = (count / len(heights)) * 100
        print(f"  {bucket:>12} : {count:3d} ({percentage:5.1f}%)")

    assert sum(area_bins.values()) == len(areas)
    assert sum(width_bins.values()) == len(widths)
    assert sum(height_bins.values()) == len(heights)