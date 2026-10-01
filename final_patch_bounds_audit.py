import cv2
import pandas as pd
from pathlib import Path


PATCH_SIZE = 192

DATASET_DIR = Path("outputs/patch_dataset")

FILES = [
    "positive_train.csv",
    "positive_validation.csv",
    "hard_negative_train.csv",
    "healthy_negative_train.csv",
]


def main():
    print("FINAL PATCH BOUNDS AUDIT")
    print("-------------------------")

    total = 0
    invalid = []

    for filename in FILES:
        csv_path = DATASET_DIR / filename
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            total += 1

            image_path = Path(row["image_path"])
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                invalid.append(
                    (
                        filename,
                        row["image_id"],
                        "image_unreadable",
                    )
                )
                continue

            height, width = image.shape

            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            valid = (
                x1 >= 0
                and y1 >= 0
                and x2 > x1
                and y2 > y1
                and x2 <= width
                and y2 <= height
                and (x2 - x1) == PATCH_SIZE
                and (y2 - y1) == PATCH_SIZE
            )

            if not valid:
                invalid.append(
                    (
                        filename,
                        row["image_id"],
                        "invalid_patch",
                        x1,
                        y1,
                        x2,
                        y2,
                        width,
                        height,
                    )
                )

    print(f"Checked patches: {total}")
    print(f"Invalid patches: {len(invalid)}")

    if invalid:
        print()
        print("Invalid examples:")
        for item in invalid[:20]:
            print(item)

    print("-------------------------")

    if invalid:
        print("FAILED")
    else:
        print("PASSED")


if __name__ == "__main__":
    main()