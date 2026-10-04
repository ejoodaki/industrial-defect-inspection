from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from final_test import load_model, load_pair, evaluate_image


PREDICTIONS_CSV = Path("outputs/final_test/final_test_predictions.csv")
TEST_DIR = Path("test")
OUTPUT_DIR = Path("outputs/final_test_error_audit")

RF_THRESHOLD = 0.30


def select_representative_errors(df):
    """Select a small, representative subset of FN/FP test errors."""

    errors = df[df["category"].isin(["FN", "FP"])].copy()

    selected = []

    # ---------------------------------------------------------
    # False Negatives
    # ---------------------------------------------------------
    fn = errors[errors["category"] == "FN"].copy()

    # 5 lowest RF scores
    selected.append(
        fn.nsmallest(5, "rf_probability")
    )

    # 5 closest to the operating threshold
    fn_near = fn.assign(
        threshold_distance=(fn["rf_probability"] - RF_THRESHOLD).abs()
    )
    selected.append(
        fn_near.nsmallest(5, "threshold_distance")
    )

    # 3 largest predicted masks
    selected.append(
        fn.nlargest(3, "predicted_pixels")
    )

    # ---------------------------------------------------------
    # False Positives
    # ---------------------------------------------------------
    fp = errors[errors["category"] == "FP"].copy()

    # 5 highest RF scores
    selected.append(
        fp.nlargest(5, "rf_probability")
    )

    # 5 closest to the operating threshold
    fp_near = fp.assign(
        threshold_distance=(fp["rf_probability"] - RF_THRESHOLD).abs()
    )
    selected.append(
        fp_near.nsmallest(5, "threshold_distance")
    )

    # 3 largest predicted masks
    selected.append(
        fp.nlargest(3, "predicted_pixels")
    )

    result = pd.concat(selected, ignore_index=True)

    # Remove duplicates while preserving the first occurrence.
    result = result.drop_duplicates(subset=["image_id"])

    # Sort for easier inspection.
    result = result.sort_values(
        ["category", "rf_probability", "image_id"]
    ).reset_index(drop=True)

    return result


def make_overlay(image, gt_mask, prediction):
    """Create a simple visualization overlay."""

    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = base.copy()

    # Ground truth: green
    overlay[gt_mask > 0] = (0, 255, 0)

    # Prediction: red
    overlay[prediction > 0] = (0, 0, 255)

    # Blend with original image.
    blended = cv2.addWeighted(base, 0.60, overlay, 0.40, 0)

    return blended


def add_label(image, text):
    """Add an English label to an image."""

    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 34),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        result,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return result


def create_panel(image, gt_mask, prediction):
    """Create a 2x2 visual audit panel."""

    original = add_label(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        "Original",
    )

    gt = add_label(
        gt_mask * 255,
        "Ground Truth",
    )
    gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

    pred = add_label(
        prediction * 255,
        "Prediction",
    )
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    overlay = add_label(
        make_overlay(image, gt_mask, prediction),
        "Overlay",
    )

    top = np.hstack([original, gt])
    bottom = np.hstack([pred, overlay])

    panel = np.vstack([top, bottom])

    return panel


def process_sample(model, row):
    image_id = int(row["image_id"])

    image_path = TEST_DIR / f"{image_id}.png"
    mask_path = TEST_DIR / f"{image_id}_GT.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    image, gt_mask = load_pair(image_path, mask_path)

    prediction, probability = evaluate_image(
        model,
        image,
    )

    category = row["category"]

    category_dir = OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    panel = create_panel(
        image,
        gt_mask,
        prediction,
    )

    output_path = category_dir / f"{image_id}_{category}.png"

    cv2.imwrite(
        str(output_path),
        panel,
    )

    return {
        "image_id": image_id,
        "category": category,
        "rf_probability": float(row["rf_probability"]),
        "predicted_pixels_csv": int(row["predicted_pixels"]),
        "predicted_pixels_recomputed": int(prediction.sum()),
        "max_probability_recomputed": float(probability.max()),
        "output_path": str(output_path),
    }


def main():
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(PREDICTIONS_CSV)

    selected = select_representative_errors(df)

    selection_path = OUTPUT_DIR / "audit_selection.csv"
    selected.to_csv(
        selection_path,
        index=False,
    )

    print("=" * 70)
    print("Final Test Error Visual Audit")
    print("=" * 70)

    print(f"Total test samples: {len(df)}")
    print(f"Total errors: {(df['category'].isin(['FN', 'FP'])).sum()}")
    print(f"Selected samples: {len(selected)}")
    print(f"Operating threshold: {RF_THRESHOLD:.2f}")
    print()

    print("Selection:")
    print(
        selected[
            [
                "image_id",
                "category",
                "rf_probability",
                "predicted_pixels",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Selection saved to: {selection_path}")
    print()

    print("Loading frozen U-Net...")
    model = load_model()

    audit_rows = []

    for _, row in selected.iterrows():
        image_id = int(row["image_id"])

        print(
            f"Auditing {image_id} "
            f"({row['category']}, "
            f"RF={row['rf_probability']:.3f})..."
        )

        result = process_sample(
            model,
            row,
        )

        audit_rows.append(result)

    results_df = pd.DataFrame(audit_rows)

    results_path = OUTPUT_DIR / "audit_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("Audit completed")
    print("=" * 70)
    print(f"Samples audited: {len(results_df)}")
    print(f"Visual results: {OUTPUT_DIR}")
    print(f"Audit results: {results_path}")
    print()
    print("Next step: visually inspect the selected FN/FP panels.")


if __name__ == "__main__":
    main()