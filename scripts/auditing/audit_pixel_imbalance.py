from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


TRAIN_CSV = Path("splits/train.csv")
VAL_CSV = Path("splits/validation.csv")


def audit_split(csv_path, name):
    data = pd.read_csv(csv_path)

    defect_ratios = []

    for _, row in data.iterrows():

        mask = np.asarray(
            Image.open(
                row["mask_path"]
            ).convert("L"),
            dtype=np.uint8,
        )

        total_pixels = mask.size

        defect_pixels = np.count_nonzero(
            mask > 0
        )

        defect_ratio = (
            defect_pixels / total_pixels
        )

        defect_ratios.append(
            defect_ratio
        )

    ratios = np.asarray(
        defect_ratios,
        dtype=np.float64,
    )

    print()
    print(f"{name} PIXEL IMBALANCE")
    print("-" * 30)

    print(
        f"Samples: {len(ratios)}"
    )

    print(
        f"Mean defect ratio:   "
        f"{ratios.mean():.8f}"
    )

    print(
        f"Median defect ratio: "
        f"{np.median(ratios):.8f}"
    )

    print(
        f"P10:                 "
        f"{np.percentile(ratios, 10):.8f}"
    )

    print(
        f"P25:                 "
        f"{np.percentile(ratios, 25):.8f}"
    )

    print(
        f"P50:                 "
        f"{np.percentile(ratios, 50):.8f}"
    )

    print(
        f"P75:                 "
        f"{np.percentile(ratios, 75):.8f}"
    )

    print(
        f"P90:                 "
        f"{np.percentile(ratios, 90):.8f}"
    )

    print(
        f"Max:                 "
        f"{ratios.max():.8f}"
    )

    print()

    print(
        f"Mean defect %:       "
        f"{ratios.mean() * 100:.4f}%"
    )

    print(
        f"Median defect %:     "
        f"{np.median(ratios) * 100:.4f}%"
    )


def main():

    print("L-01 PIXEL IMBALANCE AUDIT")
    print("==========================")

    audit_split(
        TRAIN_CSV,
        "TRAIN",
    )

    audit_split(
        VAL_CSV,
        "VALIDATION",
    )

    print()
    print("PIXEL IMBALANCE AUDIT COMPLETED")


if __name__ == "__main__":
    main()