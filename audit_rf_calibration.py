from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.calibration import calibration_curve


VAL_FEATURES = Path(
    "outputs/segmentation_feature_audit/segmentation_features.csv"
)

VAL_SPLIT = Path(
    "splits/validation.csv"
)

MODEL_PATH = Path(
    "outputs/decision_layer/random_forest_classifier.joblib"
)

OUTPUT_PATH = Path(
    "outputs/decision_layer/random_forest_calibration_audit.csv"
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

N_BINS = 10


def main():

    print("RF CALIBRATION AUDIT")
    print("====================")
    print()

    for path in [
        VAL_FEATURES,
        VAL_SPLIT,
        MODEL_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    features = pd.read_csv(
        VAL_FEATURES
    )

    split = pd.read_csv(
        VAL_SPLIT
    )

    data = split.merge(
        features,
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(split):
        raise ValueError(
            "Validation merge lost samples."
        )

    y_true = data[
        "has_defect"
    ].astype(int).to_numpy()

    x = data[
        FEATURE_COLUMNS
    ]

    model = load(
        MODEL_PATH
    )

    probabilities = model.predict_proba(
        x
    )[:, 1]

    # --------------------------------------------------
    # Calibration bins
    # --------------------------------------------------

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=N_BINS,
            strategy="uniform",
        )
    )

    # calibration_curve may omit empty bins,
    # so build the bins explicitly as well.
    bin_edges = np.linspace(
        0.0,
        1.0,
        N_BINS + 1,
    )

    rows = []

    for i in range(N_BINS):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == N_BINS - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "lower": lower,
                    "upper": upper,
                    "samples": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_defect_rate": np.nan,
                    "calibration_gap": np.nan,
                }
            )
            continue

        mean_probability = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        calibration_gap = (
            observed_rate
            - mean_probability
        )

        rows.append(
            {
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "samples": count,
                "mean_predicted_probability":
                    mean_probability,
                "observed_defect_rate":
                    observed_rate,
                "calibration_gap":
                    calibration_gap,
            }
        )

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    valid = results[
        results["samples"] > 0
    ]

    weighted_gap = (
        np.average(
            valid["calibration_gap"].abs(),
            weights=valid["samples"],
        )
    )

    print(
        f"Validation samples: {len(data)}"
    )

    print()
    print("CALIBRATION BINS")
    print("----------------")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"Weighted mean absolute "
        f"calibration gap: {weighted_gap:.6f}"
    )

    print()
    print("Saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()