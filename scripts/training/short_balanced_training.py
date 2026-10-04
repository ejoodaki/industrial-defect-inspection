from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import KolektorDataset
from unet import UNet
from losses import BCEDiceLoss


TRAIN_CSV = Path("splits/train.csv")

TARGET_SIZE = 384
BATCH_SIZE = 2

NUM_ITERATIONS = 200
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


def dice_score(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    intersection = (
        predictions * targets
    ).sum()

    denominator = (
        predictions.sum()
        + targets.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    ).item()


def main():
    print("L-01 SHORT BALANCED TRAINING EXPERIMENT")
    print("=======================================")

    print(f"Device:          {DEVICE}")
    print(f"Target size:     {TARGET_SIZE}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Iterations:      {NUM_ITERATIONS}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print()

    dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    data = pd.read_csv(TRAIN_CSV)

    healthy_count = int(
        (data["has_defect"] == 0).sum()
    )

    defective_count = int(
        (data["has_defect"] == 1).sum()
    )

    weights = []

    for label in data["has_defect"]:
        if label == 0:
            weights.append(1.0 / healthy_count)
        else:
            weights.append(1.0 / defective_count)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=NUM_ITERATIONS * BATCH_SIZE,
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    model = UNet().to(DEVICE)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    print("TRAINING")
    print("--------")

    for iteration, (images, masks) in enumerate(loader, start=1):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            probabilities = torch.sigmoid(logits)

            max_probability = probabilities.max().item()

            dice = dice_score(
                logits,
                masks,
            )

        if (
            iteration == 1
            or iteration % 25 == 0
            or iteration == NUM_ITERATIONS
        ):
            print(
                f"Iteration {iteration:3d} | "
                f"loss={loss.item():.6f} | "
                f"dice={dice:.6f} | "
                f"max_prob={max_probability:.6f}"
            )

        if iteration >= NUM_ITERATIONS:
            break

    print()
    print("FINAL RESULT")
    print("------------")

    print(
        "Short balanced training experiment completed."
    )

    print()
    print(
        "IMPORTANT: This is a diagnostic experiment, "
        "not final model training."
    )


if __name__ == "__main__":
    main()