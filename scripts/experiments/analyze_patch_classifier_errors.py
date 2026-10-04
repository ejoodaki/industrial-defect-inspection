
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from patch_classifier import PatchClassifier
from patch_classifier_dataset import PatchClassifierDataset


CHECKPOINT = Path(
    "outputs/patch_classifier/classifier_short_best.pt"
)

VALIDATION_CSVS = [
    "outputs/patch_dataset/positive_validation.csv",
    "outputs/patch_dataset/healthy_negative_validation.csv",
    "outputs/patch_dataset/hard_negative_validation.csv",
]

BATCH_SIZE = 16
THRESHOLD = 0.5


def main():
    print("PATCH CLASSIFIER ERROR ANALYSIS")
    print("--------------------------------")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    dataset = PatchClassifierDataset(
        VALIDATION_CSVS
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cpu")

    model = PatchClassifier().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    rows = []

    dataset_index = 0

    with torch.no_grad():
        for patches, labels in loader:
            batch_size = patches.shape[0]

            patches = patches.to(device)

            logits = model(patches)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            predictions = (
                probabilities >= THRESHOLD
            ).long()

            labels = labels.squeeze(1).long()

            for batch_index in range(batch_size):
                sample = dataset.samples[
                    dataset_index + batch_index
                ]

                rows.append(
                    {
                        "dataset_index": (
                            dataset_index
                            + batch_index
                        ),
                        "image_id": str(
                            sample["image_id"]
                        ),
                        "patch_type": str(
                            sample["patch_type"]
                        ),
                        "image_path": str(
                            sample["image_path"]
                        ),
                        "mask_path": str(
                            sample["mask_path"]
                        ),
                        "label": int(
                            labels[
                                batch_index
                            ].item()
                        ),
                        "prediction": int(
                            predictions[
                                batch_index
                            ].item()
                        ),
                        "probability": float(
                            probabilities[
                                batch_index
                            ].item()
                        ),
                    }
                )

            dataset_index += batch_size

    results = pd.DataFrame(rows)

    if len(results) != len(dataset):
        raise RuntimeError(
            "Prediction count does not match "
            "dataset size: "
            f"{len(results)} vs {len(dataset)}"
        )

    tp = int(
        (
            (results["label"] == 1)
            & (results["prediction"] == 1)
        ).sum()
    )

    tn = int(
        (
            (results["label"] == 0)
            & (results["prediction"] == 0)
        ).sum()
    )

    fp = int(
        (
            (results["label"] == 0)
            & (results["prediction"] == 1)
        ).sum()
    )

    fn = int(
        (
            (results["label"] == 1)
            & (results["prediction"] == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if fn + tp > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(results)
        if len(results) > 0
        else 0.0
    )

    print()
    print("Validation samples:", len(results))

    print()
    print("CONFUSION MATRIX")
    print("----------------")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    print()
    print("METRICS")
    print("-------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")

    print()
    print("ERROR COUNTS")
    print("------------")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    output_dir = Path(
        "outputs/patch_classifier/error_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "validation_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # Save explicit error tables.
    false_negatives = results[
        (results["label"] == 1)
        & (results["prediction"] == 0)
    ].copy()

    false_positives = results[
        (results["label"] == 0)
        & (results["prediction"] == 1)
    ].copy()

    false_negatives_path = (
        output_dir
        / "false_negatives.csv"
    )

    false_positives_path = (
        output_dir
        / "false_positives.csv"
    )

    false_negatives.to_csv(
        false_negatives_path,
        index=False,
    )

    false_positives.to_csv(
        false_positives_path,
        index=False,
    )

    print()
    print("Saved:")
    print(results_path)
    print(false_negatives_path)
    print(false_positives_path)

    print()
    print(
        "PATCH CLASSIFIER ERROR ANALYSIS PASSED"
    )


if __name__ == "__main__":
    main()
