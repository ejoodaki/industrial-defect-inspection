from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from final_test import (
    TEST_DIR,
    load_model,
    load_pair,
    evaluate_image,
)


PREDICTIONS_CSV = Path(
    "outputs/final_test/final_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/final_test_error_audit"
)

OUTPUT_CSV = OUTPUT_DIR / "all_test_errors_quantitative.csv"
OUTPUT_TXT = OUTPUT_DIR / "all_test_errors_quantitative.txt"


def connected_components(mask):
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_width": 0,
            "largest_component_height": 0,
        }

    component_stats = stats[1:]

    largest_index = np.argmax(
        component_stats[:, cv2.CC_STAT_AREA]
    )

    largest = component_stats[largest_index]

    return {
        "component_count": int(count - 1),
        "largest_component_area": int(
            largest[cv2.CC_STAT_AREA]
        ),
        "largest_component_width": int(
            largest[cv2.CC_STAT_WIDTH]
        ),
        "largest_component_height": int(
            largest[cv2.CC_STAT_HEIGHT]
        ),
    }


def bbox_features(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return {
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_area": 0,
            "bbox_aspect_ratio": 0.0,
        }

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "bbox_aspect_ratio": (
            width / height if height > 0 else 0.0
        ),
    }


def calculate_mask_metrics(gt_mask, prediction):
    gt = gt_mask > 0
    pred = prediction > 0

    gt_pixels = int(gt.sum())
    prediction_pixels = int(pred.sum())

    intersection = int(
        np.logical_and(gt, pred).sum()
    )

    union = int(
        np.logical_or(gt, pred).sum()
    )

    false_positive_pixels = int(
        np.logical_and(~gt, pred).sum()
    )

    false_negative_pixels = int(
        np.logical_and(gt, ~pred).sum()
    )

    true_negative_pixels = int(
        np.logical_and(~gt, ~pred).sum()
    )

    pixel_precision = (
        intersection / prediction_pixels
        if prediction_pixels > 0
        else 0.0
    )

    pixel_recall = (
        intersection / gt_pixels
        if gt_pixels > 0
        else 0.0
    )

    pixel_iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    pixel_dice = (
        2.0 * intersection
        / (gt_pixels + prediction_pixels)
        if (gt_pixels + prediction_pixels) > 0
        else 0.0
    )

    image_area = gt.shape[0] * gt.shape[1]

    gt_density = (
        gt_pixels / image_area
        if image_area > 0
        else 0.0
    )

    prediction_density = (
        prediction_pixels / image_area
        if image_area > 0
        else 0.0
    )

    return {
        "gt_pixels": gt_pixels,
        "prediction_pixels": prediction_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "false_positive_pixels": false_positive_pixels,
        "false_negative_pixels": false_negative_pixels,
        "true_negative_pixels": true_negative_pixels,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_iou": pixel_iou,
        "pixel_dice": pixel_dice,
        "gt_density": gt_density,
        "prediction_density": prediction_density,
    }


def diagnose(category, metrics):
    gt_pixels = metrics["gt_pixels"]
    prediction_pixels = metrics["prediction_pixels"]
    precision = metrics["pixel_precision"]
    recall = metrics["pixel_recall"]
    dice = metrics["pixel_dice"]

    # FP: healthy ground truth but segmentation activated.
    if category == "FP" and gt_pixels == 0:
        if prediction_pixels > 0:
            return "Segmentation false activation on healthy sample"

        return "No segmentation activation"

    # FN: defect exists but segmentation predicts nothing.
    if category == "FN":
        if gt_pixels > 0 and prediction_pixels == 0:
            return "Complete segmentation miss"

        if recall < 0.20 and precision < 0.20:
            return "Strong segmentation failure"

        if recall < 0.20 and precision >= 0.20:
            return "Strong under-segmentation"

        if recall >= 0.50 and precision < 0.20:
            return "Strong over-segmentation"

        if dice < 0.30:
            return "Weak segmentation overlap"

        if dice >= 0.50:
            return "Reasonable segmentation; inspect decision layer"

        return "Mixed segmentation quality"

    return "No clear diagnosis"


def main():
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {PREDICTIONS_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.read_csv(
        PREDICTIONS_CSV
    )

    errors_df = predictions_df[
        predictions_df["category"].isin(
            ["FN", "FP"]
        )
    ].copy()

    errors_df = errors_df.sort_values(
        by=["category", "image_id"]
    ).reset_index(drop=True)

    print("=" * 78)
    print("Full Final-Test Error Root-Cause Audit")
    print("=" * 78)
    print(
        f"Total test samples: "
        f"{len(predictions_df)}"
    )
    print(
        f"Total errors: "
        f"{len(errors_df)}"
    )
    print(
        f"FN: "
        f"{(errors_df['category'] == 'FN').sum()}"
    )
    print(
        f"FP: "
        f"{(errors_df['category'] == 'FP').sum()}"
    )
    print()

    print("Loading frozen U-Net...")
    model = load_model()
    print("Model loaded.")
    print()

    results = []

    for index, row in errors_df.iterrows():

        image_id = int(row["image_id"])
        category = row["category"]
        rf_probability = float(
            row["rf_probability"]
        )

        image_path = (
            TEST_DIR / f"{image_id}.png"
        )

        mask_path = (
            TEST_DIR / f"{image_id}_GT.png"
        )

        image, gt_mask = load_pair(
            image_path,
            mask_path,
        )

        prediction, probability = evaluate_image(
            model,
            image,
        )

        metrics = calculate_mask_metrics(
            gt_mask,
            prediction,
        )

        gt_components = connected_components(
            gt_mask
        )

        prediction_components = connected_components(
            prediction
        )

        gt_bbox = bbox_features(
            gt_mask
        )

        prediction_bbox = bbox_features(
            prediction
        )

        diagnosis = diagnose(
            category,
            metrics,
        )

        result = {
            "image_id": image_id,
            "category": category,
            "rf_probability": rf_probability,
            "max_probability": float(
                probability.max()
            ),
            **metrics,

            "gt_component_count":
                gt_components["component_count"],

            "gt_largest_component_area":
                gt_components[
                    "largest_component_area"
                ],

            "gt_largest_component_width":
                gt_components[
                    "largest_component_width"
                ],

            "gt_largest_component_height":
                gt_components[
                    "largest_component_height"
                ],

            "gt_bbox_width":
                gt_bbox["bbox_width"],

            "gt_bbox_height":
                gt_bbox["bbox_height"],

            "gt_bbox_area":
                gt_bbox["bbox_area"],

            "gt_bbox_aspect_ratio":
                gt_bbox["bbox_aspect_ratio"],

            "prediction_component_count":
                prediction_components[
                    "component_count"
                ],

            "prediction_largest_component_area":
                prediction_components[
                    "largest_component_area"
                ],

            "prediction_largest_component_width":
                prediction_components[
                    "largest_component_width"
                ],

            "prediction_largest_component_height":
                prediction_components[
                    "largest_component_height"
                ],

            "prediction_bbox_width":
                prediction_bbox["bbox_width"],

            "prediction_bbox_height":
                prediction_bbox["bbox_height"],

            "prediction_bbox_area":
                prediction_bbox["bbox_area"],

            "prediction_bbox_aspect_ratio":
                prediction_bbox[
                    "bbox_aspect_ratio"
                ],

            "diagnosis": diagnosis,
        }

        results.append(result)

        print(
            f"[{index + 1:03d}/{len(errors_df):03d}] "
            f"{image_id}: "
            f"{category} | "
            f"RF={rf_probability:.3f} | "
            f"IoU={metrics['pixel_iou']:.3f} | "
            f"Dice={metrics['pixel_dice']:.3f} | "
            f"{diagnosis}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    diagnosis_counts = (
        results_df["diagnosis"]
        .value_counts()
    )

    category_diagnosis = pd.crosstab(
        results_df["category"],
        results_df["diagnosis"],
    )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Full Final-Test Error Root-Cause Audit\n"
        )
        f.write("=" * 78 + "\n\n")

        f.write(
            f"Total test samples: "
            f"{len(predictions_df)}\n"
        )

        f.write(
            f"Total errors: "
            f"{len(results_df)}\n"
        )

        f.write(
            f"FN: "
            f"{(results_df['category'] == 'FN').sum()}\n"
        )

        f.write(
            f"FP: "
            f"{(results_df['category'] == 'FP').sum()}\n\n"
        )

        f.write("Diagnosis counts\n")
        f.write("-" * 78 + "\n")

        for diagnosis, count in diagnosis_counts.items():
            percentage = (
                100.0 * count / len(results_df)
            )

            f.write(
                f"{diagnosis}: "
                f"{count} "
                f"({percentage:.1f}%)\n"
            )

        f.write("\n")
        f.write("Diagnosis by category\n")
        f.write("-" * 78 + "\n")

        f.write(
            category_diagnosis.to_string()
        )

        f.write("\n\n")
        f.write("FN diagnosis breakdown\n")
        f.write("-" * 78 + "\n")

        fn_df = results_df[
            results_df["category"] == "FN"
        ]

        fn_counts = (
            fn_df["diagnosis"]
            .value_counts()
        )

        for diagnosis, count in fn_counts.items():
            percentage = (
                100.0 * count / len(fn_df)
            )

            f.write(
                f"{diagnosis}: "
                f"{count} "
                f"({percentage:.1f}%)\n"
            )

        f.write("\n")
        f.write("FP diagnosis breakdown\n")
        f.write("-" * 78 + "\n")

        fp_df = results_df[
            results_df["category"] == "FP"
        ]

        fp_counts = (
            fp_df["diagnosis"]
            .value_counts()
        )

        for diagnosis, count in fp_counts.items():
            percentage = (
                100.0 * count / len(fp_df)
            )

            f.write(
                f"{diagnosis}: "
                f"{count} "
                f"({percentage:.1f}%)\n"
            )

    print()
    print("=" * 78)
    print("Full audit completed")
    print("=" * 78)
    print(f"CSV report: {OUTPUT_CSV}")
    print(f"Text report: {OUTPUT_TXT}")
    print()

    print("Diagnosis summary:")
    print(
        diagnosis_counts.to_string()
    )

    print()
    print("Diagnosis by category:")
    print(
        category_diagnosis.to_string()
    )


if __name__ == "__main__":
    main()