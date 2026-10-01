from pathlib import Path

import cv2
import pandas as pd


PATCH_SIZE = 192


def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def get_defect_bbox(mask):
    ys, xs = __import__("numpy").where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def crop_centered(image, center_x, center_y, patch_size=PATCH_SIZE):
    height, width = image.shape

    half = patch_size // 2

    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))

    x1 = max(0, min(x1, width - patch_size))
    y1 = max(0, min(y1, height - patch_size))

    x2 = x1 + patch_size
    y2 = y1 + patch_size

    if x2 > width or y2 > height:
        return None

    return image[y1:y2, x1:x2]


def make_positive_patch(image, mask):
    bbox = get_defect_bbox(mask)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return crop_centered(
        image,
        center_x,
        center_y,
    )


def build_split(split_path):
    manifest = pd.read_csv(split_path)

    samples = []

    for _, row in manifest.iterrows():

        if int(row["has_defect"]) != 1:
            continue

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)

        patch = make_positive_patch(image, mask)

        if patch is None:
            raise ValueError(
                f"Could not create positive patch for {row['image_id']}"
            )

        samples.append(
            {
                "image_id": str(row["image_id"]),
                "label": 1,
                "patch_type": "positive",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    return samples


def main():
    train_path = Path("splits/train.csv")
    validation_path = Path("splits/validation.csv")

    train_samples = build_split(train_path)
    validation_samples = build_split(validation_path)

    output_dir = Path("outputs/patch_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_samples).to_csv(
        output_dir / "positive_train.csv",
        index=False,
    )

    pd.DataFrame(validation_samples).to_csv(
        output_dir / "positive_validation.csv",
        index=False,
    )

    print(
        f"Saved train artifact: "
        f"{output_dir / 'positive_train.csv'}"
    )

    print(
        f"Saved validation artifact: "
        f"{output_dir / 'positive_validation.csv'}"
    )

    print()

    train_ids = {sample["image_id"] for sample in train_samples}
    validation_ids = {
        sample["image_id"] for sample in validation_samples
    }

    overlap = train_ids & validation_ids

    print("PATCH DATASET SPLIT AUDIT")
    print("-------------------------")
    print(f"Train positive samples:       {len(train_samples)}")
    print(f"Validation positive samples:  {len(validation_samples)}")
    print(f"Cross-split overlap:           {len(overlap)}")

    print()

    assert len(train_samples) == 197
    assert len(validation_samples) == 49
    assert len(overlap) == 0

    print("POSITIVE PATCH SPLIT PASSED")


if __name__ == "__main__":
    main()