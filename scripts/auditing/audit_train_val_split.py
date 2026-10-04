from pathlib import Path

import cv2
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
SPLIT_DIR = DATASET_DIR / "splits"

TRAIN_SPLIT = SPLIT_DIR / "train.csv"
VAL_SPLIT = SPLIT_DIR / "validation.csv"


REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "mask_path",
    "has_defect",
}


def validate_split(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{name} split not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{name} split is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} split missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df["image_id"].duplicated().any():
        duplicates = df.loc[
            df["image_id"].duplicated(),
            "image_id"
        ].tolist()

        raise ValueError(
            f"Duplicate IDs in {name}: {duplicates}"
        )

    healthy = int((df["has_defect"] == 0).sum())
    defective = int((df["has_defect"] == 1).sum())

    unreadable_images = 0
    unreadable_masks = 0
    dimension_mismatches = 0
    missing_images = 0
    missing_masks = 0

    for _, row in df.iterrows():
        image_path = DATASET_DIR / row["image_path"]
        mask_path = DATASET_DIR / row["mask_path"]

        if not image_path.exists():
            missing_images += 1
            continue

        if not mask_path.exists():
            missing_masks += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            unreadable_images += 1
            continue

        if mask is None:
            unreadable_masks += 1
            continue

        if image.shape != mask.shape:
            dimension_mismatches += 1

    print(f"--- {name.upper()} ---")
    print(f"Samples              : {len(df)}")
    print(f"Healthy              : {healthy}")
    print(f"Defective            : {defective}")
    print(f"Missing images       : {missing_images}")
    print(f"Missing masks        : {missing_masks}")
    print(f"Unreadable images    : {unreadable_images}")
    print(f"Unreadable masks     : {unreadable_masks}")
    print(f"Dimension mismatches : {dimension_mismatches}")

    if any(
        [
            missing_images,
            missing_masks,
            unreadable_images,
            unreadable_masks,
            dimension_mismatches,
        ]
    ):
        raise ValueError(f"{name} split validation failed.")

    print("Status               : PASSED")
    print()

    return df


def main():
    train_df = validate_split("train", TRAIN_SPLIT)
    val_df = validate_split("validation", VAL_SPLIT)

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])

    overlap = train_ids & val_ids

    print("=== CROSS-SPLIT AUDIT ===")
    print(f"Train samples        : {len(train_ids)}")
    print(f"Validation samples   : {len(val_ids)}")
    print(f"Overlapping IDs      : {len(overlap)}")

    if overlap:
        print(f"Overlap examples     : {sorted(overlap)[:10]}")
        raise ValueError("Train/validation overlap detected.")

    print("Overlap check        : PASSED")
    print("PASSED")


if __name__ == "__main__":
    main()