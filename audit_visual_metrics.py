from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from final_test import load_model, load_pair, evaluate_image


PREDICTIONS_CSV = Path(
    "outputs/final_test/final_test_predictions.csv"
)

SELECTION_CSV = Path(
    "outputs/final_test_error_audit/audit_selection.csv"
)

OUTPUT_DIR = Path(
    "outputs/final_test_error_audit"
)

TEST_DIR = Path("test")


def connected_components(mask):
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_width": 0,
            "largest_component_height": 0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_index = 1 + int(np.argmax(areas))

    return {
        "component_count": int(num_labels - 1),
        "largest_component_area": int(
            stats[largest_index, cv2.CC_STAT_AREA]
        ),
        "largest_component_width": int(
            stats[largest_index, cv2.CC_STAT_WIDTH]
        ),
        "largest_component_height": int(
            stats[largest_index, cv2.CC_STAT_HEIGHT]
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
        "bbox_aspect_ratio": width / height if height else 0.0,
    }


def calculate_mask_metrics(gt, prediction):
    gt = (gt > 0).astype(np.uint8)
    prediction = (prediction > 0).astype(np.uint8)

    gt_pixels = int(gt.sum())
    prediction_pixels = int(prediction.sum())

    intersection = int(
        np.logical_and(gt > 0, prediction > 0).sum()
    )

    union = int(
        np.logical_or(gt > 0, prediction > 0).sum()
    )

    false_positive_pixels = int(
        np.logical_and(gt == 0, prediction > 0).sum()
    )

    false_negative_pixels = int(
        np.logical_and(gt > 0, prediction == 0).sum()
    )

    true_negative_pixels = int(
        np.logical_and(gt == 0, prediction == 0).sum()
    )

    gt_recall = (
        intersection / gt_pixels
        if gt_pixels > 0
        else 0.0
    )

    prediction_precision = (
        intersection / prediction_pixels
        if prediction_pixels > 0
        else 0.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    dice = (
        2.0 * intersection
        / (gt_pixels + prediction_pixels)
        if (gt_pixels + prediction_pixels) > 0
        else 1.0
    )

    image_area = gt.shape[0] * gt.shape[1]

    gt_density = gt_pixels / image_area
    prediction_density = prediction_pixels / image_area

    return {
        "gt_pixels": gt_pixels,
        "prediction_pixels": prediction_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "false_positive_pixels": false_positive_pixels,
        "false_negative_pixels": false_negative_pixels,
        "true_negative_pixels": true_negative_pixels,
        "pixel_recall": gt_recall,
        "pixel_precision": prediction_precision,
        "pixel_iou": iou,
        "pixel_dice": dice,
        "gt_density": gt_density,
        "prediction_density_recomputed": prediction_density,
    }


def diagnose(metrics):
    gt_pixels = metrics["gt_pixels"]
    prediction_pixels = metrics["prediction_pixels"]
    pixel_recall = metrics["pixel_recall"]
    pixel_precision = metrics["pixel_precision"]
    dice = metrics["pixel_dice"]

    # Healthy ground truth + prediction
    if gt_pixels == 0 and prediction_pixels > 0:
        return "Segmentation false activation on healthy sample"

    # Defect exists but segmentation barely overlaps it
    if gt_pixels > 0 and prediction_pixels > 0:
        if pixel_recall < 0.20 and pixel_precision < 0.20:
            return "Strong segmentation failure"

        if pixel_recall < 0.20 and pixel_precision >= 0.20:
            return "Strong under-segmentation"

        if pixel_recall >= 0.50 and pixel_precision < 0.20:
            return "Strong over-segmentation"

        if dice < 0.30:
            return "Weak segmentation overlap"

        if dice >= 0.50:
            return "Reasonable segmentation; inspect decision layer"

        return "Mixed segmentation quality"

    # Defect exists but model predicts nothing
    if gt_pixels > 0 and prediction_pixels == 0:
        return "Complete segmentation miss"

    return "No clear diagnosis"


def main():
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {PREDICTIONS_CSV}"
        )

    if not SELECTION_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {SELECTION_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = pd.read_csv(SELECTION_CSV)

    print("=" * 78)
    print("Quantitative Final Test Visual Audit")
    print("=" * 78)
    print(f"Selected samples: {len(selected)}")
    print()

    print("Loading frozen U-Net...")
    model = load_model()
    print("Model loaded.")
    print()

    results = []

    for _, row in selected.iterrows():

        image_id = int(row["image_id"])
        category = row["category"]
        rf_probability = float(row["rf_probability"])

        image_path = TEST_DIR / f"{image_id}.png"
        mask_path = TEST_DIR / f"{image_id}_GT.png"

        image, gt_mask = load_pair(
            image_path,
            mask_path,
        )

        prediction, probability = evaluate_image(
            model,
            image,
        )

        mask_metrics = calculate_mask_metrics(
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
            mask_metrics
        )

        result = {
            "image_id": image_id,
            "category": category,
            "rf_probability": rf_probability,
            "max_probability": float(
                probability.max()
            ),
            **mask_metrics,

            "gt_component_count":
                gt_components["component_count"],

            "gt_largest_component_area":
                gt_components["largest_component_area"],

            "gt_largest_component_width":
                gt_components["largest_component_width"],

            "gt_largest_component_height":
                gt_components["largest_component_height"],

            "gt_bbox_width":
                gt_bbox["bbox_width"],

            "gt_bbox_height":
                gt_bbox["bbox_height"],

            "gt_bbox_area":
                gt_bbox["bbox_area"],

            "gt_bbox_aspect_ratio":
                gt_bbox["bbox_aspect_ratio"],

            "prediction_component_count":
                prediction_components["component_count"],

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
                prediction_bbox["bbox_aspect_ratio"],

            "diagnosis": diagnosis,
        }

        results.append(result)

        print(
            f"{image_id}: "
            f"{category} | "
            f"RF={rf_probability:.3f} | "
            f"GT={mask_metrics['gt_pixels']} | "
            f"Pred={mask_metrics['prediction_pixels']} | "
            f"IoU={mask_metrics['pixel_iou']:.3f} | "
            f"Dice={mask_metrics['pixel_dice']:.3f} | "
            f"{diagnosis}"
        )

    results_df = pd.DataFrame(results)

    csv_path = (
        OUTPUT_DIR
        / "quantitative_visual_audit.csv"
    )

    txt_path = (
        OUTPUT_DIR
        / "quantitative_visual_audit.txt"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # Human-readable report
    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Quantitative Final Test Visual Audit\n"
        )
        f.write("=" * 78 + "\n\n")

        for _, row in results_df.iterrows():

            f.write(
                f"Image: {int(row['image_id'])}\n"
            )

            f.write(
                f"Category: {row['category']}\n"
            )

            f.write(
                f"RF score: "
                f"{row['rf_probability']:.3f}\n"
            )

            f.write(
                f"GT pixels: "
                f"{int(row['gt_pixels'])}\n"
            )

            f.write(
                f"Prediction pixels: "
                f"{int(row['prediction_pixels'])}\n"
            )

            f.write(
                f"Intersection: "
                f"{int(row['intersection_pixels'])}\n"
            )

            f.write(
                f"Pixel precision: "
                f"{row['pixel_precision']:.4f}\n"
            )

            f.write(
                f"Pixel recall: "
                f"{row['pixel_recall']:.4f}\n"
            )

            f.write(
                f"Pixel IoU: "
                f"{row['pixel_iou']:.4f}\n"
            )

            f.write(
                f"Pixel Dice: "
                f"{row['pixel_dice']:.4f}\n"
            )

            f.write(
                f"GT components: "
                f"{int(row['gt_component_count'])}\n"
            )

            f.write(
                f"Prediction components: "
                f"{int(row['prediction_component_count'])}\n"
            )

            f.write(
                f"GT bbox: "
                f"{int(row['gt_bbox_width'])} x "
                f"{int(row['gt_bbox_height'])}\n"
            )

            f.write(
                f"Prediction bbox: "
                f"{int(row['prediction_bbox_width'])} x "
                f"{int(row['prediction_bbox_height'])}\n"
            )

            f.write(
                f"Diagnosis: "
                f"{row['diagnosis']}\n"
            )

            f.write("-" * 78 + "\n\n")

        f.write("\nSummary\n")
        f.write("=" * 78 + "\n")

        f.write(
            f"Samples: {len(results_df)}\n"
        )

        f.write(
            f"FN samples: "
            f"{(results_df['category'] == 'FN').sum()}\n"
        )

        f.write(
            f"FP samples: "
            f"{(results_df['category'] == 'FP').sum()}\n"
        )

        f.write("\nDiagnosis counts:\n")

        diagnosis_counts = (
            results_df["diagnosis"]
            .value_counts()
        )

        for diagnosis, count in diagnosis_counts.items():
            f.write(
                f"  {diagnosis}: {count}\n"
            )

    print()
    print("=" * 78)
    print("Audit completed")
    print("=" * 78)
    print(f"CSV report: {csv_path}")
    print(f"Text report: {txt_path}")
    print()

    print("Diagnosis summary:")
    print(
        results_df["diagnosis"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()