from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATASET_DIR / "train" / "train_manifest.csv"
OUTPUT_DIR = DATASET_DIR / "splits"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def create_split():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "image_id",
        "image_path",
        "mask_path",
        "has_defect",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Manifest is empty.")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected.")

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["has_defect"],
    )

    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    val_df = val_df.sort_values("image_id").reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "validation.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Verify there is no overlap.
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])

    overlap = train_ids & val_ids

    if overlap:
        raise ValueError(
            f"Train/validation overlap detected: {sorted(overlap)}"
        )

    # Verify that every manifest sample is assigned exactly once.
    all_ids = train_ids | val_ids
    manifest_ids = set(df["image_id"])

    if all_ids != manifest_ids:
        raise ValueError("Some samples were lost during splitting.")

    print("=== TRAIN / VALIDATION SPLIT ===")
    print(f"Total samples      : {len(df)}")
    print(f"Training samples   : {len(train_df)}")
    print(f"Validation samples : {len(val_df)}")
    print()

    print("Training distribution:")
    print(
        f"  Healthy          : "
        f"{(train_df['has_defect'] == 0).sum()}"
    )
    print(
        f"  Defective        : "
        f"{(train_df['has_defect'] == 1).sum()}"
    )
    print()

    print("Validation distribution:")
    print(
        f"  Healthy          : "
        f"{(val_df['has_defect'] == 0).sum()}"
    )
    print(
        f"  Defective        : "
        f"{(val_df['has_defect'] == 1).sum()}"
    )
    print()

    print(f"Random state       : {RANDOM_STATE}")
    print(f"Validation ratio   : {VALIDATION_SIZE}")
    print(f"Train output       : {train_path}")
    print(f"Validation output  : {val_path}")
    print("Overlap check      : PASSED")
    print("Coverage check     : PASSED")
    print("PASSED")


if __name__ == "__main__":
    create_split()