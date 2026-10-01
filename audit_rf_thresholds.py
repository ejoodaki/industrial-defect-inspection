from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    "outputs/decision_layer"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "random_forest_threshold_audit.csv"
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

THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05,
)


def main():

    print("RF THRESHOLD AUDIT")
    print("==================")
    print()

    # --------------------------------------------------
    # Validate inputs
    # --------------------------------------------------

    for path in [
        VAL_FEATURES,
        VAL_SPLIT,
        MODEL_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------
    # Load validation data
    # --------------------------------------------------

    features = pd.read_csv(
        VAL_FEATURES
    )

    split = pd.read_csv(
        VAL_SPLIT
    )

    if len(features) != len(split):
        raise ValueError(
            "Validation feature count does not "
            "match validation split count."
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

    y_true = data["has_defect"].astype(int)

    x = data[
        FEATURE_COLUMNS
    ]

    # --------------------------------------------------
    # Load existing RF
    # --------------------------------------------------

    model = load(
        MODEL_PATH
    )

    probabilities = (
        model.predict_proba(x)[:, 1]
    )

    print(
        f"Validation samples: {len(data)}"
    )

    print(
        f"Defective samples:   "
        f"{int(y_true.sum())}"
    )

    print(
        f"Healthy samples:     "
        f"{int((y_true == 0).sum())}"
    )

    print()

    # --------------------------------------------------
    # ROC / PR metrics
    # --------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    average_precision = (
        average_precision_score(
            y_true,
            probabilities,
        )
    )

    print(
        f"ROC-AUC:             {roc_auc:.6f}"
    )

    print(
        f"Average Precision:   "
        f"{average_precision:.6f}"
    )

    print()

    # --------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------

    rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "fpr": fp / (fp + tn),
            }
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print("THRESHOLD RESULTS")
    print("-----------------")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "Saved:"
    )
    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()