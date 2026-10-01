import csv
from pathlib import Path

import numpy as np
import pandas as pd

import full_validation_feature_audit as base


OUTPUT_DIR = Path(
    "outputs/final_test_audit/candidate_aggregation"
)

RESULTS_CSV = (
    OUTPUT_DIR / "candidate_aggregation_comparison.csv"
)

REPORT_TXT = (
    OUTPUT_DIR / "candidate_aggregation_report.txt"
)


def evaluate_with_mode(
    model,
    image,
    top_k=None,
):
    all_candidates = []

    for patch_size in base.PATCH_SIZES:
        candidates = base.generate_candidates(
            image,
            patch_size,
            base.STRIDE,
        )

        all_candidates.extend(candidates)

    all_candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    if top_k is None:
        selected = all_candidates
    else:
        selected = all_candidates[:top_k]

    full_probability = np.zeros(
        image.shape,
        dtype=np.float32,
    )

    for candidate in selected:
        x1 = candidate["x1"]
        y1 = candidate["y1"]
        x2 = candidate["x2"]
        y2 = candidate["y2"]

        patch = image[
            y1:y2,
            x1:x2,
        ]

        probability = base.predict_candidate(
            model,
            patch,
        )

        probability = base.restore_prediction(
            probability,
            patch.shape,
        )

        full_probability[
            y1:y2,
            x1:x2
        ] = np.maximum(
            full_probability[
                y1:y2,
                x1:x2
            ],
            probability,
        )

    prediction = (
        full_probability >= base.THRESHOLD
    ).astype(np.uint8)

    return prediction, full_probability


def calculate_metrics(
    prediction,
    mask,
):
    prediction = prediction.astype(bool)
    mask = mask.astype(bool)

    tp = int(
        np.logical_and(
            prediction,
            mask,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction,
            ~mask,
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~prediction,
            mask,
        ).sum()
    )

    tn = int(
        np.logical_and(
            ~prediction,
            ~mask,
        ).sum()
    )

    predicted_pixels = int(
        prediction.sum()
    )

    gt_pixels = int(
        mask.sum()
    )

    intersection = tp

    union = int(
        np.logical_or(
            prediction,
            mask,
        ).sum()
    )

    dice_denominator = (
        predicted_pixels + gt_pixels
    )

    dice = (
        (2.0 * intersection / dice_denominator)
        if dice_denominator > 0
        else 1.0
    )

    recall = (
        tp / gt_pixels
        if gt_pixels > 0
        else 0.0
    )

    precision = (
        tp / predicted_pixels
        if predicted_pixels > 0
        else 0.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    image_area = mask.shape[0] * mask.shape[1]

    density = (
        predicted_pixels / image_area
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_pixels": gt_pixels,
        "predicted_pixels": predicted_pixels,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_iou": iou,
        "pixel_dice": dice,
        "prediction_density": density,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("VALIDATION CANDIDATE AGGREGATION AUDIT")
    print("=" * 72)

    print()
    print("Loading frozen U-Net...")
    model = base.load_model()
    print("Model loaded.")
    print()

    with open(
        base.VAL_CSV,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    defective = [
        row
        for row in rows
        if row["has_defect"] == "1"
    ][:base.MAX_DEFECTIVE]

    healthy = [
        row
        for row in rows
        if row["has_defect"] == "0"
    ][:base.MAX_HEALTHY]

    evaluation_rows = (
        defective + healthy
    )

    print(
        f"Validation samples: "
        f"{len(evaluation_rows)}"
    )

    print(
        f"Defective: "
        f"{len(defective)}"
    )

    print(
        f"Healthy: "
        f"{len(healthy)}"
    )

    print()

    results = []

    for index, row in enumerate(
        evaluation_rows,
        start=1,
    ):
        image_id = Path(
            row["image_path"]
        ).stem

        image, mask = base.load_pair(
            row["image_path"],
            row["mask_path"],
        )

        print(
            f"[{index:03d}/"
            f"{len(evaluation_rows):03d}] "
            f"{image_id}"
        )

        for mode, top_k in [
            ("ALL", None),
            ("TOP_5", base.TOP_K),
        ]:
            prediction, probability = (
                evaluate_with_mode(
                    model,
                    image,
                    top_k=top_k,
                )
            )

            metrics = calculate_metrics(
                prediction,
                mask,
            )

            has_defect = (
                int(row["has_defect"])
            )

            if has_defect == 1:
                image_prediction = (
                    "TP"
                    if metrics["predicted_pixels"] > 0
                    else "FN"
                )
            else:
                image_prediction = (
                    "FP"
                    if metrics["predicted_pixels"] > 0
                    else "TN"
                )

            results.append(
                {
                    "image_id": image_id,
                    "has_defect": has_defect,
                    "mode": mode,
                    "image_prediction":
                        image_prediction,
                    "max_probability":
                        float(
                            probability.max()
                        ),
                    **metrics,
                }
            )

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    summary_rows = []

    for mode in [
        "ALL",
        "TOP_5",
    ]:
        current = df[
            df["mode"] == mode
        ]

        defective_current = current[
            current["has_defect"] == 1
        ]

        healthy_current = current[
            current["has_defect"] == 0
        ]

        tp = (
            (
                defective_current[
                    "image_prediction"
                ] == "TP"
            ).sum()
        )

        fn = (
            (
                defective_current[
                    "image_prediction"
                ] == "FN"
            ).sum()
        )

        tn = (
            (
                healthy_current[
                    "image_prediction"
                ] == "TN"
            ).sum()
        )

        fp = (
            (
                healthy_current[
                    "image_prediction"
                ] == "FP"
            ).sum()
        )

        defect_recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        healthy_specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        summary_rows.append(
            {
                "mode": mode,
                "TP": tp,
                "FN": fn,
                "TN": tn,
                "FP": fp,
                "defect_recall":
                    defect_recall,
                "healthy_specificity":
                    healthy_specificity,
                "mean_dice":
                    current[
                        "pixel_dice"
                    ].mean(),
                "median_dice":
                    current[
                        "pixel_dice"
                    ].median(),
                "mean_prediction_density":
                    current[
                        "prediction_density"
                    ].mean(),
                "median_prediction_density":
                    current[
                        "prediction_density"
                    ].median(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "VALIDATION CANDIDATE "
            "AGGREGATION AUDIT\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(
            "Purpose:\n"
            "Compare the frozen baseline using ALL "
            "candidates against TOP-5 candidates.\n\n"
        )

        f.write(
            "No test data was used.\n"
        )
        f.write(
            "No model was retrained.\n"
        )
        f.write(
            "No segmentation threshold was tuned.\n\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )

        f.write("\n\n")

        all_row = summary[
            summary["mode"] == "ALL"
        ].iloc[0]

        top5_row = summary[
            summary["mode"] == "TOP_5"
        ].iloc[0]

        fp_change = (
            top5_row["FP"]
            - all_row["FP"]
        )

        fn_change = (
            top5_row["FN"]
            - all_row["FN"]
        )

        f.write(
            "Interpretation\n"
        )
        f.write("-" * 72 + "\n")

        f.write(
            f"FP change (TOP_5 - ALL): "
            f"{fp_change:+.0f}\n"
        )

        f.write(
            f"FN change (TOP_5 - ALL): "
            f"{fn_change:+.0f}\n"
        )

        f.write(
            f"Defect recall change: "
            f"{top5_row['defect_recall'] - all_row['defect_recall']:+.4f}\n"
        )

        f.write(
            "\nThis audit is descriptive and "
            "validation-only.\n"
        )

    print()
    print("=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)

    print(
        f"CSV: {RESULTS_CSV}"
    )

    print(
        f"Report: {REPORT_TXT}"
    )


if __name__ == "__main__":
    main()