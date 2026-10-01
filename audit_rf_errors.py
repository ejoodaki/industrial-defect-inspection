from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


VAL_FEATURES = Path(
    "outputs/segmentation_feature_audit/segmentation_features.csv"
)

VAL_SPLIT = Path(
    "splits/validation.csv"
)

MODEL_PATH = Path(
    "outputs/decision_layer/random_forest_classifier.joblib"
)

OUTPUT_DIR = Path(
    "outputs/rf_error_audit"
)

FEATURE_COLUMNS = [
    "predicted_pixels",
    "prediction_density",
    "max_probability",
    "mean_probability",
    "component_count",
    "largest_component_area",
    "largest_component_bbox_area",
    "largest_component_width",
    "largest_component_height",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
]


def load_data():
    features = pd.read_csv(
        VAL_FEATURES
    )

    split = pd.read_csv(
        VAL_SPLIT
    )

    merged = split[
        ["image_id", "image_path", "mask_path", "has_defect"]
    ].merge(
        features[
            ["image_id", *FEATURE_COLUMNS]
        ],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(split):
        raise RuntimeError(
            "Validation split and feature CSV do not match."
        )

    return merged


def make_overlay(
    image,
    mask,
    prediction,
):
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = image_bgr.copy()

    gt_region = mask > 0
    pred_region = prediction > 0

    # Ground truth: green
    overlay[gt_region] = (
        0.5 * overlay[gt_region]
        + 0.5 * np.array(
            [0, 255, 0],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    # Prediction: red
    overlay[pred_region] = (
        0.5 * overlay[pred_region]
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

    model = joblib.load(
        MODEL_PATH
    )

    data = load_data()

    probabilities = model.predict_proba(
        data[FEATURE_COLUMNS]
    )[:, 1]

    data["probability"] = probabilities
    data["prediction"] = (
        probabilities >= 0.5
    ).astype(int)

    errors = data[
        data["prediction"]
        != data["has_defect"]
    ].copy()

    print("=" * 60)
    print("RANDOM FOREST ERROR VISUAL AUDIT")
    print("=" * 60)
    print()

    print(
        f"Validation samples: {len(data)}"
    )
    print(
        f"False negatives: "
        f"{((data.has_defect == 1) & (data.prediction == 0)).sum()}"
    )
    print(
        f"False positives: "
        f"{((data.has_defect == 0) & (data.prediction == 1)).sum()}"
    )
    print()

    print(
        "Generating error visualizations..."
    )
    print()

    for index, row in enumerate(
        errors.itertuples(index=False),
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

        # Recreate the RF-level binary decision
        # as an image-level label.
        #
        # The segmentation mask itself is reconstructed
        # from existing U-Net prediction artifacts only
        # when available. For this first audit we save
        # image + GT to establish the error set.
        #
        # A full prediction reconstruction is handled
        # in the next controlled step.

        category = (
            "FN"
            if row.has_defect == 1
            else "FP"
        )

        # Save original image.
        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{category}_{row.image_id}_image.png"
            ),
            image,
        )

        # Save GT mask.
        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{category}_{row.image_id}_gt.png"
            ),
            mask * 255,
        )

        print(
            f"{index}/{len(errors)} | "
            f"{category} | "
            f"{row.image_id} | "
            f"RF probability={row.probability:.3f}"
        )

    error_csv = (
        OUTPUT_DIR
        / "rf_errors.csv"
    )

    errors[
        [
            "image_id",
            "has_defect",
            "prediction",
            "probability",
            *FEATURE_COLUMNS,
        ]
    ].to_csv(
        error_csv,
        index=False,
    )

    print()
    print(
        f"Saved error metadata: {error_csv}"
    )
    print(
        f"Saved error images: {OUTPUT_DIR}"
    )
    print()
    print(
        "RANDOM FOREST ERROR AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()