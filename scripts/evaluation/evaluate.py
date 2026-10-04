from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from unet import UNet


DEVICE = torch.device("cpu")

VAL_CSV = "splits/validation.csv"
CHECKPOINT_PATH = "outputs/checkpoints/baseline_best.pt"

TARGET_SIZE = 384
BATCH_SIZE = 2
THRESHOLD = 0.5


def calculate_metrics(prediction, target):
    """
    Calculate binary segmentation metrics.

    prediction: binary tensor (B, 1, H, W)
    target:     binary tensor (B, 1, H, W)
    """

    prediction = prediction.bool()
    target = target.bool()

    true_positive = (prediction & target).sum().item()
    false_positive = (prediction & ~target).sum().item()
    false_negative = (~prediction & target).sum().item()

    intersection = true_positive
    union = (
        true_positive
        + false_positive
        + false_negative
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        )

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    if true_positive + false_positive == 0:
        precision = 1.0 if true_positive + false_negative == 0 else 0.0
    else:
        precision = (
            true_positive
            / (true_positive + false_positive)
        )

    if true_positive + false_negative == 0:
        recall = 1.0
    else:
        recall = (
            true_positive
            / (true_positive + false_negative)
        )

    return (
        dice,
        iou,
        precision,
        recall,
    )


def main():
    print()
    print("L-01 BASELINE VALIDATION EVALUATION")
    print("===================================")

    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Threshold:  {THRESHOLD}")

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

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print(f"Validation samples: {len(dataset)}")
    print(f"Checkpoint epoch:   {checkpoint['epoch']}")
    print(
        f"Checkpoint val loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_samples = 0
    healthy_samples = 0
    defective_samples = 0

    healthy_false_positive = 0
    defective_detected = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= THRESHOLD
            )

            targets = masks >= 0.5

            for index in range(images.size(0)):
                prediction = predictions[index:index + 1]
                target = targets[index:index + 1]

                tp = (
                    prediction & target
                ).sum().item()

                fp = (
                    prediction & ~target
                ).sum().item()

                fn = (
                    ~prediction & target
                ).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn

                target_has_defect = (
                    target.sum().item() > 0
                )

                prediction_has_defect = (
                    prediction.sum().item() > 0
                )

                total_samples += 1

                if target_has_defect:
                    defective_samples += 1

                    if prediction_has_defect:
                        defective_detected += 1

                else:
                    healthy_samples += 1

                    if prediction_has_defect:
                        healthy_false_positive += 1

    dice_denominator = (
        2 * total_tp
        + total_fp
        + total_fn
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * total_tp
            / dice_denominator
        )

    union = (
        total_tp
        + total_fp
        + total_fn
    )

    if union == 0:
        iou = 1.0
    else:
        iou = total_tp / union

    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = (
            total_tp
            / (total_tp + total_fp)
        )

    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = (
            total_tp
            / (total_tp + total_fn)
        )

    print()
    print("SEGMENTATION METRICS")
    print("--------------------")
    print(f"Dice:      {dice:.6f}")
    print(f"IoU:       {iou:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")

    print()
    print("IMAGE-LEVEL BEHAVIOR")
    print("--------------------")
    print(f"Total samples:          {total_samples}")
    print(f"Healthy samples:        {healthy_samples}")
    print(f"Defective samples:      {defective_samples}")
    print(
        f"Healthy false positives: "
        f"{healthy_false_positive}"
    )
    print(
        f"Defective images detected: "
        f"{defective_detected}"
    )

    if defective_samples > 0:
        detection_rate = (
            defective_detected
            / defective_samples
        )
    else:
        detection_rate = 0.0

    if healthy_samples > 0:
        false_positive_rate = (
            healthy_false_positive
            / healthy_samples
        )
    else:
        false_positive_rate = 0.0

    print(
        f"Defective detection rate: "
        f"{detection_rate:.6f}"
    )

    print(
        f"Healthy false-positive rate: "
        f"{false_positive_rate:.6f}"
    )

    print()
    print("VALIDATION EVALUATION COMPLETED")


if __name__ == "__main__":
    main()