from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_FEATURES = Path(
    "outputs/train_segmentation_feature_audit/segmentation_features.csv"
)

VAL_FEATURES = Path(
    "outputs/segmentation_feature_audit/segmentation_features.csv"
)

TRAIN_SPLIT = Path("splits/train.csv")
VAL_SPLIT = Path("splits/validation.csv")

OUTPUT_DIR = Path(
    "outputs/decision_layer"
)

MODEL_PATH = OUTPUT_DIR / "logistic_regression_pipeline.joblib"
METRICS_PATH = OUTPUT_DIR / "validation_metrics.csv"

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


def load_data(
    feature_path,
    split_path,
):
    features = pd.read_csv(feature_path)
    split = pd.read_csv(split_path)

    required_feature_columns = [
        "image_id",
        *FEATURE_COLUMNS,
    ]

    missing_features = [
        column
        for column in required_feature_columns
        if column not in features.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing feature columns: {missing_features}"
        )

    if "image_id" not in split.columns:
        raise RuntimeError(
            f"Missing image_id column in {split_path}"
        )

    if "has_defect" not in split.columns:
        raise RuntimeError(
            f"Missing has_defect column in {split_path}"
        )

    merged = split[
        ["image_id", "has_defect"]
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
            f"Feature/split mismatch: "
            f"{split_path} has {len(split)} rows, "
            f"merged result has {len(merged)}"
        )

    return merged


def build_model():
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(
    model,
    x,
    y,
):
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "samples": len(y),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "fpr": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return metrics


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("DECISION LAYER — LOGISTIC REGRESSION")
    print("=" * 60)
    print()

    train = load_data(
        TRAIN_FEATURES,
        TRAIN_SPLIT,
    )

    validation = load_data(
        VAL_FEATURES,
        VAL_SPLIT,
    )

    print(
        f"Train samples: {len(train)}"
    )
    print(
        f"Validation samples: {len(validation)}"
    )
    print()

    x_train = train[
        FEATURE_COLUMNS
    ]
    y_train = train[
        "has_defect"
    ].astype(int)

    x_val = validation[
        FEATURE_COLUMNS
    ]
    y_val = validation[
        "has_defect"
    ].astype(int)

    print("Train label distribution:")
    print(
        y_train.value_counts()
        .sort_index()
    )
    print()

    print("Validation label distribution:")
    print(
        y_val.value_counts()
        .sort_index()
    )
    print()

    model = build_model()

    print(
        "Training logistic regression..."
    )

    model.fit(
        x_train,
        y_train,
    )

    metrics = evaluate(
        model,
        x_val,
        y_val,
    )

    print()
    print(
        "=" * 60
    )
    print(
        "VALIDATION RESULTS"
    )
    print(
        "=" * 60
    )

    for key, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    try:
        import joblib

        joblib.dump(
            model,
            MODEL_PATH,
        )
    except ImportError as exc:
        raise RuntimeError(
            "joblib is required to save the model."
        ) from exc

    print()
    print(
        f"Model saved: {MODEL_PATH}"
    )
    print(
        f"Metrics saved: {METRICS_PATH}"
    )
    print()
    print(
        "DECISION LAYER COMPLETE"
    )


if __name__ == "__main__":
    main()