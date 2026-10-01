from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PATCH_SIZE = 192
SAMPLES_PER_TYPE = 5

OUTPUT_DIR = Path("outputs/patch_audit")


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


def make_positive_coords(mask):
    bbox = get_defect_bbox(mask)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    px1 = int(round(center_x - PATCH_SIZE / 2))
    py1 = int(round(center_y - PATCH_SIZE / 2))

    height, width = mask.shape

    px1 = max(0, min(px1, width - PATCH_SIZE))
    py1 = max(0, min(py1, height - PATCH_SIZE))

    return (
        px1,
        py1,
        px1 + PATCH_SIZE,
        py1 + PATCH_SIZE,
    )


def save_audit_image(
    image,
    mask,
    coords,
    output_path,
):
    x1, y1, x2, y2 = coords

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = image_rgb.copy()

    defect_pixels = mask > 0

    overlay[defect_pixels] = (
        0.5 * overlay[defect_pixels]
        + 0.5 * np.array(
            [0, 0, 255],
            dtype=np.float64,
        )
    ).astype(np.uint8)

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 0),
        2,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        overlay,
    )


def audit_positive_samples(manifest):
    defective = manifest[
        manifest["has_defect"] == 1
    ]

    results = []

    for _, row in defective.head(
        SAMPLES_PER_TYPE
    ).iterrows():

        image = load_grayscale(
            row["image_path"]
        )

        mask = load_grayscale(
            row["mask_path"]
        )

        coords = make_positive_coords(mask)

        if coords is None:
            raise ValueError(
                f"No bbox for {row['image_id']}"
            )

        x1, y1, x2, y2 = coords

        patch_mask = mask[
            y1:y2,
            x1:x2,
        ]

        defect_pixels = int(
            np.count_nonzero(patch_mask > 0)
        )

        if defect_pixels == 0:
            raise AssertionError(
                f"Positive patch is empty: "
                f"{row['image_id']}"
            )

        results.append(
            (
                str(row["image_id"]),
                "positive",
                defect_pixels,
                coords,
            )
        )

        save_audit_image(
            image,
            mask,
            coords,
            OUTPUT_DIR
            / f"{row['image_id']}_positive.png",
        )

    return results


def audit_healthy_samples(manifest):
    healthy = manifest[
        manifest["has_defect"] == 0
    ]

    results = []

    for _, row in healthy.head(
        SAMPLES_PER_TYPE
    ).iterrows():

        image = load_grayscale(
            row["image_path"]
        )

        mask = load_grayscale(
            row["mask_path"]
        )

        height, width = mask.shape

        coords = (
            0,
            0,
            PATCH_SIZE,
            PATCH_SIZE,
        )

        if height < PATCH_SIZE or width < PATCH_SIZE:
            raise ValueError(
                f"Image too small: "
                f"{row['image_id']}"
            )

        x1, y1, x2, y2 = coords

        patch_mask = mask[
            y1:y2,
            x1:x2,
        ]

        defect_pixels = int(
            np.count_nonzero(patch_mask > 0)
        )

        if defect_pixels != 0:
            raise AssertionError(
                f"Healthy patch contains defect: "
                f"{row['image_id']}"
            )

        results.append(
            (
                str(row["image_id"]),
                "healthy_negative",
                defect_pixels,
                coords,
            )
        )

        save_audit_image(
            image,
            mask,
            coords,
            OUTPUT_DIR
            / f"{row['image_id']}_healthy_negative.png",
        )

    return results


def audit_hard_negative_samples():
    hard_negative_path = Path(
        "outputs/patch_dataset/hard_negative_train.csv"
    )

    if not hard_negative_path.exists():
        raise FileNotFoundError(
            "Expected hard-negative CSV was not found: "
            f"{hard_negative_path}"
        )

    data = pd.read_csv(
        hard_negative_path
    )

    if data.empty:
        raise ValueError(
            "Hard-negative CSV is empty."
        )

    sample = data.head(
        SAMPLES_PER_TYPE
    )

    results = []

    for _, row in sample.iterrows():

        image = load_grayscale(
            row["image_path"]
        )

        mask = load_grayscale(
            row["mask_path"]
        )

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        patch_mask = mask[
            y1:y2,
            x1:x2,
        ]

        defect_pixels = int(
            np.count_nonzero(patch_mask > 0)
        )

        if defect_pixels != 0:
            raise AssertionError(
                f"Hard negative contains defect: "
                f"{row['image_id']}"
            )

        results.append(
            (
                str(row["image_id"]),
                "hard_negative",
                defect_pixels,
                (x1, y1, x2, y2),
            )
        )

        save_audit_image(
            image,
            mask,
            (x1, y1, x2, y2),
            OUTPUT_DIR
            / f"{row['image_id']}_hard_negative.png",
        )

    return results


def main():
    train_manifest = pd.read_csv(
        "splits/train.csv"
    )

    positive_results = audit_positive_samples(
        train_manifest
    )

    healthy_results = audit_healthy_samples(
        train_manifest
    )

    hard_results = audit_hard_negative_samples()

    all_results = (
        positive_results
        + healthy_results
        + hard_results
    )

    print("PATCH INTEGRITY AUDIT")
    print("---------------------")

    for (
        image_id,
        patch_type,
        defect_pixels,
        coords,
    ) in all_results:

        print(
            f"{patch_type:18s} "
            f"{image_id:>6s} "
            f"defect_pixels={defect_pixels:6d} "
            f"coords={coords}"
        )

    print()
    print(
        f"Audited patches: {len(all_results)}"
    )

    assert len(positive_results) == 5
    assert len(healthy_results) == 5
    assert len(hard_results) == 5

    assert all(
        pixels > 0
        for _, _, pixels, _ in positive_results
    )

    assert all(
        pixels == 0
        for _, _, pixels, _ in healthy_results
    )

    assert all(
        pixels == 0
        for _, _, pixels, _ in hard_results
    )

    print()
    print("PATCH INTEGRITY AUDIT PASSED")
    print(
        f"Visual audit files: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()