from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from train_feature_audit import (
    load_model,
    evaluate_image,
)


RF_ERRORS = Path(
    "outputs/rf_error_audit/rf_errors.csv"
)

VAL_SPLIT = Path(
    "splits/validation.csv"
)

OUTPUT_DIR = Path(
    "outputs/rf_prediction_audit"
)

DEVICE = torch.device("cpu")


def load_validation_data():
    errors = pd.read_csv(
        RF_ERRORS
    )

    validation = pd.read_csv(
        VAL_SPLIT
    )

    merged = errors[
        ["image_id", "has_defect", "prediction", "probability"]
    ].merge(
        validation[
            [
                "image_id",
                "image_path",
                "mask_path",
            ]
        ],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(errors):
        raise RuntimeError(
            "RF error metadata and validation split do not match."
        )

    return merged


def make_overlay(
    image,
    prediction,
    mask,
):
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = image_bgr.copy()

    gt_region = mask > 0
    prediction_region = prediction > 0

    if gt_region.any():
        overlay[gt_region] = (
            0.5 * overlay[gt_region]
            + 0.5 * np.array(
                [0, 255, 0],
                dtype=np.float32,
            )
        ).astype(np.uint8)

    if prediction_region.any():
        overlay[prediction_region] = (
            0.5 * overlay[prediction_region]
            + 0.5 * np.array(
                [0, 0, 255],
                dtype=np.float32,
            )
        ).astype(np.uint8)

    return overlay


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model()

    data = load_validation_data()

    print("=" * 60)
    print("RF ERROR -> U-NET PREDICTION AUDIT")
    print("=" * 60)
    print(
        f"RF error samples: {len(data)}"
    )
    print()

    results = []

    for index, row in enumerate(
        data.itertuples(index=False),
        start=1,
    ):
        image_path = Path(
            row.image_path
        )

        mask_path = Path(
            row.mask_path
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        mask = (
            mask > 127
        ).astype(np.uint8)

        prediction, probability = (
        evaluate_image(
        model,
        image,
        )
    )

        predicted_pixels = int(
            prediction.sum()
        )

        gt_pixels = int(
            mask.sum()
        )

        intersection = int(
            np.logical_and(
                prediction > 0,
                mask > 0,
            ).sum()
        )

        union = int(
            np.logical_or(
                prediction > 0,
                mask > 0,
            ).sum()
        )

        dice = (
            2.0 * intersection
            / (predicted_pixels + gt_pixels)
            if (predicted_pixels + gt_pixels) > 0
            else 1.0
        )

        iou = (
            intersection / union
            if union > 0
            else 1.0
        )

        category = (
            "FN"
            if row.has_defect == 1
            else "FP"
        )

        overlay = make_overlay(
            image,
            prediction,
            mask,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{category}_{row.image_id}_image.png"
            ),
            image,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{category}_{row.image_id}_gt.png"
            ),
            mask * 255,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{category}_{row.image_id}_prediction.png"
            ),
            prediction * 255,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{category}_{row.image_id}_overlay.png"
            ),
            overlay,
        )

        results.append(
            {
                "image_id": row.image_id,
                "category": category,
                "rf_probability": float(
                    row.probability
                ),
                "u_net_predicted_pixels":
                    predicted_pixels,
                "gt_pixels": gt_pixels,
                "dice": float(dice),
                "iou": float(iou),
            }
        )

        print(
            f"{index}/{len(data)} | "
            f"{category} | "
            f"{row.image_id} | "
            f"RF={row.probability:.3f} | "
            f"U-Net px={predicted_pixels} | "
            f"Dice={dice:.4f}"
        )

    results_csv = (
        OUTPUT_DIR
        / "rf_prediction_audit.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        results_csv,
        index=False,
    )

    print()
    print(
        f"Saved results: {results_csv}"
    )
    print(
        f"Saved visualizations: {OUTPUT_DIR}"
    )
    print()
    print(
        "RF ERROR -> U-NET PREDICTION AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()