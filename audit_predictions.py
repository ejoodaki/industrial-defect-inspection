from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from unet import UNet


CHECKPOINT = Path("outputs/checkpoints/baseline_best.pt")
VAL_CSV = Path("splits/validation.csv")

BATCH_SIZE = 2
TARGET_SIZE = 384

DEVICE = torch.device("cpu")


def main():
    print("L-01 RAW PREDICTION AUDIT")
    print("==========================")

    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print()

    dataset = KolektorDataset(
        VAL_CSV,
        target_size=TARGET_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    healthy_max_probs = []
    defective_max_probs = []

    healthy_mean_probs = []
    defective_mean_probs = []

    healthy_positive_pixels = []
    defective_positive_pixels = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            for i in range(images.size(0)):

                prob = probabilities[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()

                max_prob = float(prob.max())
                mean_prob = float(prob.mean())

                positive_pixels = int(
                    (prob >= 0.5).sum()
                )

                is_defective = bool(
                    (mask > 0.5).any()
                )

                if is_defective:
                    defective_max_probs.append(max_prob)
                    defective_mean_probs.append(mean_prob)
                    defective_positive_pixels.append(
                        positive_pixels
                    )
                else:
                    healthy_max_probs.append(max_prob)
                    healthy_mean_probs.append(mean_prob)
                    healthy_positive_pixels.append(
                        positive_pixels
                    )

    print("DATASET SUMMARY")
    print("---------------")
    print(f"Healthy samples:   {len(healthy_max_probs)}")
    print(f"Defective samples: {len(defective_max_probs)}")
    print()

    print("MAX PROBABILITY")
    print("----------------")

    print("Healthy:")
    print(
        f"  min:    {np.min(healthy_max_probs):.6f}"
    )
    print(
        f"  median: {np.median(healthy_max_probs):.6f}"
    )
    print(
        f"  mean:   {np.mean(healthy_max_probs):.6f}"
    )
    print(
        f"  max:    {np.max(healthy_max_probs):.6f}"
    )
    print()

    print("Defective:")
    print(
        f"  min:    {np.min(defective_max_probs):.6f}"
    )
    print(
        f"  median: {np.median(defective_max_probs):.6f}"
    )
    print(
        f"  mean:   {np.mean(defective_max_probs):.6f}"
    )
    print(
        f"  max:    {np.max(defective_max_probs):.6f}"
    )
    print()

    print("MEAN PROBABILITY")
    print("----------------")

    print("Healthy:")
    print(
        f"  mean:   {np.mean(healthy_mean_probs):.6f}"
    )
    print(
        f"  max:    {np.max(healthy_mean_probs):.6f}"
    )
    print()

    print("Defective:")
    print(
        f"  mean:   {np.mean(defective_mean_probs):.6f}"
    )
    print(
        f"  max:    {np.max(defective_mean_probs):.6f}"
    )
    print()

    print("POSITIVE PIXELS @ THRESHOLD 0.5")
    print("--------------------------------")

    print("Healthy:")
    print(
        f"  samples with positive pixels: "
        f"{sum(x > 0 for x in healthy_positive_pixels)}"
    )
    print(
        f"  maximum positive pixels: "
        f"{max(healthy_positive_pixels)}"
    )
    print()

    print("Defective:")
    print(
        f"  samples with positive pixels: "
        f"{sum(x > 0 for x in defective_positive_pixels)}"
    )
    print(
        f"  maximum positive pixels: "
        f"{max(defective_positive_pixels)}"
    )
    print()

    print("THRESHOLD SWEEP")
    print("---------------")

    all_results = []

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    for threshold in thresholds:

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for prob, is_defective in zip(
            healthy_max_probs,
            [False] * len(healthy_max_probs),
        ):
            predicted_defect = prob >= threshold

            if predicted_defect:
                fp += 1
            else:
                tn += 1

        for prob, is_defective in zip(
            defective_max_probs,
            [True] * len(defective_max_probs),
        ):
            predicted_defect = prob >= threshold

            if predicted_defect:
                tp += 1
            else:
                fn += 1

        detection_rate = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        all_results.append(
            (
                threshold,
                tp,
                fn,
                fp,
                tn,
                detection_rate,
                false_positive_rate,
            )
        )

    print(
        "Threshold | TP | FN | FP | TN | "
        "Detection | False Positive"
    )

    print(
        "----------|----|----|----|----|"
        "-----------|----------------"
    )

    for result in all_results:

        (
            threshold,
            tp,
            fn,
            fp,
            tn,
            detection_rate,
            false_positive_rate,
        ) = result

        print(
            f"{threshold:9.2f} | "
            f"{tp:2d} | "
            f"{fn:2d} | "
            f"{fp:2d} | "
            f"{tn:3d} | "
            f"{detection_rate:9.4f} | "
            f"{false_positive_rate:14.4f}"
        )

    print()
    print("RAW PREDICTION AUDIT COMPLETED")


if __name__ == "__main__":
    main()