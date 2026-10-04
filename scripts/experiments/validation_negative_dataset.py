from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PATCH_SIZE = 192
CANDIDATES_PER_IMAGE = 100
MAX_NEGATIVES_PER_IMAGE = 2
SEED = 42


def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


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


def sample_random_patch(image, patch_size, rng):
    height, width = image.shape

    if height < patch_size or width < patch_size:
        return None

    x1 = int(rng.integers(0, width - patch_size + 1))
    y1 = int(rng.integers(0, height - patch_size + 1))

    return (
        x1,
        y1,
        x1 + patch_size,
        y1 + patch_size,
    )


def bbox_distance(patch_bbox, defect_bbox):
    px1, py1, px2, py2 = patch_bbox
    dx1, dy1, dx2, dy2 = defect_bbox

    if px2 < dx1:
        dx = dx1 - px2
    elif dx2 < px1:
        dx = px1 - dx2
    else:
        dx = 0

    if py2 < dy1:
        dy = dy1 - py2
    elif dy2 < py1:
        dy = py1 - dy2
    else:
        dy = 0

    return float(np.hypot(dx, dy))


def build_healthy_negatives(manifest, rng):
    samples = []

    healthy = manifest[manifest["has_defect"].astype(int) == 0]

    for _, row in healthy.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)

        selected = 0

        for _ in range(CANDIDATES_PER_IMAGE):
            patch_bbox = sample_random_patch(
                image,
                PATCH_SIZE,
                rng,
            )

            if patch_bbox is None:
                continue

            x1, y1, x2, y2 = patch_bbox
            mask_patch = mask[y1:y2, x1:x2]

            if np.any(mask_patch > 0):
                continue

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

            selected += 1

            if selected >= MAX_NEGATIVES_PER_IMAGE:
                break

    return samples


def build_hard_negatives(manifest, rng):
    samples = []

    defective = manifest[manifest["has_defect"].astype(int) == 1]

    for _, row in defective.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)

        defect_bbox = get_defect_bbox(mask)

        if defect_bbox is None:
            continue

        candidates = []

        for _ in range(CANDIDATES_PER_IMAGE):
            patch_bbox = sample_random_patch(
                image,
                PATCH_SIZE,
                rng,
            )

            if patch_bbox is None:
                continue

            x1, y1, x2, y2 = patch_bbox
            mask_patch = mask[y1:y2, x1:x2]

            if np.any(mask_patch > 0):
                continue

            distance = bbox_distance(
                patch_bbox,
                defect_bbox,
            )

            candidates.append(
                (
                    distance,
                    patch_bbox,
                )
            )

        candidates.sort(key=lambda item: item[0])

        for distance, patch_bbox in candidates[
            :MAX_NEGATIVES_PER_IMAGE
        ]:
            x1, y1, x2, y2 = patch_bbox

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
                    "distance_to_defect": distance,
                }
            )

    return samples


def main():
    validation_path = Path("splits/validation.csv")

    output_dir = Path("outputs/patch_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(validation_path)

    rng = np.random.default_rng(SEED)

    healthy_samples = build_healthy_negatives(
        manifest,
        rng,
    )

    hard_samples = build_hard_negatives(
        manifest,
        rng,
    )

    healthy_output = (
        output_dir / "healthy_negative_validation.csv"
    )

    hard_output = (
        output_dir / "hard_negative_validation.csv"
    )

    pd.DataFrame(healthy_samples).to_csv(
        healthy_output,
        index=False,
    )

    pd.DataFrame(hard_samples).to_csv(
        hard_output,
        index=False,
    )

    print("VALIDATION NEGATIVE DATASET")
    print("---------------------------")
    print(
        f"Healthy negatives:     {len(healthy_samples)}"
    )
    print(
        f"Hard negatives:        {len(hard_samples)}"
    )
    print()
    print(f"Saved: {healthy_output}")
    print(f"Saved: {hard_output}")

    assert len(healthy_samples) > 0
    assert len(hard_samples) > 0

    healthy_ids = {
        sample["image_id"]
        for sample in healthy_samples
    }

    hard_ids = {
        sample["image_id"]
        for sample in hard_samples
    }

    validation_ids = set(
        manifest["image_id"].astype(str)
    )

    assert healthy_ids <= validation_ids
    assert hard_ids <= validation_ids

    print()
    print("VALIDATION NEGATIVE DATASET PASSED")


if __name__ == "__main__":
    main()