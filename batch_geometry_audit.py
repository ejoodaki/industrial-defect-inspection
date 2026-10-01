from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
VAL_CSV = ROOT / "splits" / "validation.csv"


TARGETS = [
    (256, 256),
    (320, 320),
    (384, 384),
]


def analyze_split(csv_path, name):
    df = pd.read_csv(csv_path)

    heights = []
    widths = []

    for _, row in df.iterrows():
        from PIL import Image

        image = Image.open(row["image_path"])

        width, height = image.size

        widths.append(width)
        heights.append(height)

    print(f"=== {name.upper()} GEOMETRY AUDIT ===")
    print(f"Samples              : {len(df)}")
    print(
        f"Original height      : "
        f"{min(heights)} - {max(heights)}"
    )
    print(
        f"Original width       : "
        f"{min(widths)} - {max(widths)}"
    )

    original_aspect_min = min(
        h / w for h, w in zip(heights, widths)
    )

    original_aspect_max = max(
        h / w for h, w in zip(heights, widths)
    )

    print(
        f"Original H/W ratio   : "
        f"{original_aspect_min:.3f} - "
        f"{original_aspect_max:.3f}"
    )

    print()

    for target_h, target_w in TARGETS:
        distortion_values = []

        for h, w in zip(heights, widths):
            original_ratio = h / w
            target_ratio = target_h / target_w

            distortion = target_ratio / original_ratio

            distortion_values.append(distortion)

        print(
            f"Direct resize {target_h}x{target_w}:"
        )

        print(
            f"  Geometry scale ratio: "
            f"{min(distortion_values):.3f} - "
            f"{max(distortion_values):.3f}"
        )

        print(
            f"  Mean ratio          : "
            f"{sum(distortion_values) / len(distortion_values):.3f}"
        )

        print()


if __name__ == "__main__":
    analyze_split(TRAIN_CSV, "train")

    print()

    analyze_split(VAL_CSV, "validation")

    print()
    print("BATCH GEOMETRY AUDIT COMPLETED")