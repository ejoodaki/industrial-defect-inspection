from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PATCH_SIZE = 192
CANDIDATES_PER_IMAGE = 20
MAX_HEALTHY_NEGATIVES_PER_IMAGE = 2


def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def get_random_patch_coordinates(
    height,
    width,
    patch_size,
    rng,
):
    if height < patch_size or width < patch_size:
        return None

    x1 = int(
        rng.integers(
            0,
            width - patch_size + 1,
        )
    )

    y1 = int(
        rng.integers(
            0,
            height - patch_size + 1,
        )
    )

    return (
        x1,
        y1,
        x1 + patch_size,
        y1 + patch_size,
    )


def is_clean_patch(mask, coords):
    x1, y1, x2, y2 = coords

    patch_mask = mask[y1:y2, x1:x2]

    return not np.any(patch_mask > 0)


def find_healthy_negatives(
    image,
    mask,
    rng,
):
    height, width = image.shape

    candidates = []

    for _ in range(CANDIDATES_PER_IMAGE):
        coords = get_random_patch_coordinates(
            height,
            width,
            PATCH_SIZE,
            rng,
        )

        if coords is None:
            continue

        if not is_clean_patch(mask, coords):
            continue

        candidates.append(coords)

    return candidates[:MAX_HEALTHY_NEGATIVES_PER_IMAGE]


def build_healthy_negative_index(manifest_path):
    manifest = pd.read_csv(manifest_path)

    rng = np.random.default_rng(42)

    samples = []

    for _, row in manifest.iterrows():

        if int(row["has_defect"]) != 0:
            continue

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)

        negatives = find_healthy_negatives(
            image,
            mask,
            rng,
        )

        for coords in negatives:
            x1, y1, x2, y2 = coords

            samples.append(
                {
                    "image_id": str(row["image_id"]),
                    "label": 0,
                    "patch_type": "healthy_negative",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return samples


def main():
    manifest_path = Path("splits/train.csv")

    samples = build_healthy_negative_index(
        manifest_path
    )

    output_path = Path(
    "outputs/patch_dataset/healthy_negative_train.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(samples).to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved artifact: {output_path}"
    )

    print()

    print("HEALTHY NEGATIVE DATASET AUDIT")
    print("------------------------------")
    print(
        f"Healthy negative samples: "
        f"{len(samples)}"
    )

    unique_images = {
        sample["image_id"]
        for sample in samples
    }

    print(
        f"Unique healthy images used: "
        f"{len(unique_images)}"
    )

    print()

    assert len(samples) > 0

    assert all(
        sample["label"] == 0
        for sample in samples
    )

    assert all(
        sample["patch_type"] == "healthy_negative"
        for sample in samples
    )

    print("HEALTHY NEGATIVE INDEX PASSED")


if __name__ == "__main__":
    main()