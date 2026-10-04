from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PATCH_SIZE = 192


def load_grayscale(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return image


def make_positive_patch(image, mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None, "EMPTY_MASK"

    # Same logic used by PatchClassifierDataset.
    center_x = float(np.median(xs))
    center_y = float(np.median(ys))

    height, width = image.shape
    half = PATCH_SIZE // 2

    patch_x1 = int(round(center_x - half))
    patch_y1 = int(round(center_y - half))

    patch_x1 = max(
        0,
        min(patch_x1, width - PATCH_SIZE),
    )

    patch_y1 = max(
        0,
        min(patch_y1, height - PATCH_SIZE),
    )

    patch_x2 = patch_x1 + PATCH_SIZE
    patch_y2 = patch_y1 + PATCH_SIZE

    patch = image[
        patch_y1:patch_y2,
        patch_x1:patch_x2,
    ]

    mask_patch = mask[
        patch_y1:patch_y2,
        patch_x1:patch_x2,
    ]

    defect_pixels = int(
        np.count_nonzero(mask_patch > 0)
    )

    if patch.shape != (
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        return {
            "patch_shape": patch.shape,
            "crop": (
                patch_x1,
                patch_y1,
                patch_x2,
                patch_y2,
            ),
        }, "BAD_SHAPE"

    if defect_pixels == 0:
        return {
            "crop": (
                patch_x1,
                patch_y1,
                patch_x2,
                patch_y2,
            ),
            "defect_pixels": defect_pixels,
        }, "NO_DEFECT_IN_PATCH"

    return {
        "crop": (
            patch_x1,
            patch_y1,
            patch_x2,
            patch_y2,
        ),
        "defect_pixels": defect_pixels,
        "original_defect_pixels": int(
            np.count_nonzero(mask > 0)
        ),
    }, "OK"


def main():
    csv_path = Path(
        "outputs/patch_dataset/positive_train.csv"
    )

    dataframe = pd.read_csv(csv_path)

    print("POSITIVE PATCH DEBUG")
    print("--------------------")
    print(
        f"Positive samples: {len(dataframe)}"
    )
    print()

    failures = []

    for _, row in dataframe.iterrows():
        image = load_grayscale(
            row["image_path"]
        )

        mask = load_grayscale(
            row["mask_path"]
        )

        result, status = make_positive_patch(
            image,
            mask,
        )

        if status != "OK":
            failures.append(
                {
                    "image_id": str(row["image_id"]),
                    "status": status,
                    "details": result,
                }
            )

    print(
        f"Failures: {len(failures)}"
    )

    for failure in failures[:20]:
        print(failure)

    print()

    if failures:
        print(
            "POSITIVE PATCH DEBUG FAILED"
        )
        raise SystemExit(1)

    print(
        "ALL POSITIVE PATCHES PASSED"
    )


if __name__ == "__main__":
    main()