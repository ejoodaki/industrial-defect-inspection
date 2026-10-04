from pathlib import Path

import cv2
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
TRAIN_DIR = DATASET_DIR / "train"
OUTPUT_PATH = TRAIN_DIR / "train_manifest.csv"


def build_manifest():
    records = []

    for image_path in sorted(TRAIN_DIR.glob("*.png")):
        # Only canonical image files are allowed:
        # 10000.png, 10001.png, ...
        if not image_path.stem.isdigit():
            continue

        image_id = image_path.stem
        mask_path = TRAIN_DIR / f"{image_id}_GT.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for image {image_path.name}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Unreadable image: {image_path.name}")

        if mask is None:
            raise ValueError(f"Unreadable mask: {mask_path.name}")

        if image.shape != mask.shape:
            raise ValueError(
                f"Dimension mismatch: {image_path.name} "
                f"{image.shape} != {mask_path.name} {mask.shape}"
            )

        has_defect = int(cv2.countNonZero(mask) > 0)

        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.relative_to(DATASET_DIR)),
                "mask_path": str(mask_path.relative_to(DATASET_DIR)),
                "has_defect": has_defect,
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Manifest is empty.")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected.")

    df = df.sort_values("image_id").reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    healthy = int((df["has_defect"] == 0).sum())
    defective = int((df["has_defect"] == 1).sum())

    print("=== TRAIN DATASET MANIFEST ===")
    print(f"Total samples     : {len(df)}")
    print(f"Healthy samples   : {healthy}")
    print(f"Defective samples : {defective}")
    print(f"Output            : {OUTPUT_PATH}")
    print("PASSED")


if __name__ == "__main__":
    build_manifest()