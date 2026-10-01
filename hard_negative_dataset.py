from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PATCH_SIZE = 192
CANDIDATES_PER_IMAGE = 100
MAX_HARD_NEGATIVES_PER_IMAGE = 2


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


def distance_to_bbox(coords, bbox):
    x1, y1, x2, y2 = coords
    bx1, by1, bx2, by2 = bbox

    horizontal = max(
        bx1 - x2,
        x1 - bx2,
        0,
    )

    vertical = max(
        by1 - y2,
        y1 - by2,
        0,
    )

    return float(
        np.sqrt(
            horizontal ** 2
            + vertical ** 2
        )
    )


def get_defect_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def find_hard_negatives(
    image,
    mask,
    rng,
):
    height, width = image.shape

    bbox = get_defect_bbox(mask)

    if bbox is None:
        return []

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

        distance = distance_to_bbox(
            coords,
            bbox,
        )

        candidates.append(
            (
                distance,
                coords,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    selected = candidates[
        :MAX_HARD_NEGATIVES_PER_IMAGE
    ]

    return [
        {
            "coords": coords,
            "distance_to_defect": distance,
        }
        for distance, coords in selected
    ]


def build_hard_negative_index(manifest_path):
    manifest = pd.read_csv(manifest_path)

    rng = np.random.default_rng(42)

    samples = []

    for _, row in manifest.iterrows():

        if int(row["has_defect"]) != 1:
            continue

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)

        hard_negatives = find_hard_negatives(
            image,
            mask,
            rng,
        )

        for item in hard_negatives:
            x1, y1, x2, y2 = item["coords"]

            samples.append(
                {
                    "image_id": str(row["image_id"]),
                    "label": 0,
                    "patch_type": "hard_negative",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "distance_to_defect": item[
                        "distance_to_defect"
                    ],
                }
            )

    return samples


def main():
    manifest_path = Path("splits/train.csv")

    samples = build_hard_negative_index(
        manifest_path
    )

    output_path = Path(
    "outputs/patch_dataset/hard_negative_train.csv"
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
    print("HARD NEGATIVE DATASET AUDIT")
    print("---------------------------")
    print(f"Hard negative samples: {len(samples)}")

    if samples:
        distances = [
            sample["distance_to_defect"]
            for sample in samples
        ]

        print(
            f"Min distance:           {min(distances):.2f}"
        )
        print(
            f"Median distance:        "
            f"{np.median(distances):.2f}"
        )
        print(
            f"Max distance:           {max(distances):.2f}"
        )

    print()

    assert len(samples) > 0
    assert all(
        sample["label"] == 0
        for sample in samples
    )
    assert all(
        sample["patch_type"] == "hard_negative"
        for sample in samples
    )

    print("HARD NEGATIVE INDEX PASSED")


if __name__ == "__main__":
    main()