from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from unet import UNet


VAL_CSV = Path("splits/validation.csv")

CHECKPOINT = Path(
    "outputs/checkpoints/short_balanced_best.pt"
)

TARGET_SIZE = 384
BATCH_SIZE = 2
THRESHOLD = 0.5

NUM_HEALTHY = 10
NUM_DEFECTIVE = 10

DEVICE = torch.device("cpu")


def dice_score(prediction, target):
    intersection = (
        prediction * target
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    ).item()


def main():

    print("L-01 SEGMENTATION SANITY AUDIT")
    print("================================")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Threshold:  {THRESHOLD}")
    print(f"Device:     {DEVICE}")
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = KolektorDataset(
        VAL_CSV,
        target_size=TARGET_SIZE,
    )

    data = pd.read_csv(VAL_CSV)

    healthy_indices = (
        data.index[
            data["has_defect"] == 0
        ].tolist()
    )

    defective_indices = (
        data.index[
            data["has_defect"] == 1
        ].tolist()
    )

    selected_indices = (
        healthy_indices[:NUM_HEALTHY]
        + defective_indices[:NUM_DEFECTIVE]
    )

    # --------------------------------------------------
    # Targeted subset
    # --------------------------------------------------

    subset = torch.utils.data.Subset(
        dataset,
        selected_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("AUDIT SET")
    print("---------")
    print(f"Healthy samples:   {NUM_HEALTHY}")
    print(f"Defective samples: {NUM_DEFECTIVE}")
    print(f"Total samples:     {len(subset)}")
    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("CHECKPOINT")
    print("----------")
    print(
        f"Iteration:       "
        f"{checkpoint['iteration']}"
    )
    print(
        f"Validation Dice: "
        f"{checkpoint['val_dice']:.6f}"
    )
    print()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total_dice = 0.0
    total_samples = 0

    healthy_total = 0
    healthy_false_positives = 0

    defective_total = 0
    defective_detected = 0

    defective_dice_total = 0.0

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            for i in range(images.size(0)):

                prediction = predictions[i]
                target = masks[i]

                image_dice = dice_score(
                    prediction,
                    target,
                )

                total_dice += image_dice
                total_samples += 1

                predicted_pixels = (
                    prediction.sum().item()
                )

                target_pixels = (
                    target.sum().item()
                )

                if target_pixels == 0:

                    healthy_total += 1

                    if predicted_pixels > 0:
                        healthy_false_positives += 1

                else:

                    defective_total += 1

                    defective_dice_total += (
                        image_dice
                    )

                    if predicted_pixels > 0:
                        defective_detected += 1

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    mean_dice = (
        total_dice / total_samples
        if total_samples > 0
        else 0.0
    )

    defective_mean_dice = (
        defective_dice_total
        / defective_total
        if defective_total > 0
        else 0.0
    )

    detection_rate = (
        defective_detected
        / defective_total
        if defective_total > 0
        else 0.0
    )

    false_positive_rate = (
        healthy_false_positives
        / healthy_total
        if healthy_total > 0
        else 0.0
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("SANITY AUDIT RESULTS")
    print("--------------------")

    print(
        f"Mean Dice:              "
        f"{mean_dice:.6f}"
    )

    print(
        f"Defective Mean Dice:    "
        f"{defective_mean_dice:.6f}"
    )

    print(
        f"Defect Detection Rate:  "
        f"{detection_rate:.4f}"
    )

    print(
        f"Healthy False Positive: "
        f"{false_positive_rate:.4f}"
    )

    print()

    print("SEGMENTATION SANITY AUDIT COMPLETED")
    print("------------------------------------")


if __name__ == "__main__":
    main()