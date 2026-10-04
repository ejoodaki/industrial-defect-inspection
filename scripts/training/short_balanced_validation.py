from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from dataset import KolektorDataset
from unet import UNet
from losses import WeightedBCEDiceLoss


TRAIN_CSV = Path("splits/train.csv")
VAL_CSV = Path("splits/validation.csv")

TARGET_SIZE = 384
BATCH_SIZE = 2

NUM_ITERATIONS = 200
LEARNING_RATE = 0.001

VAL_HEALTHY = 10
VAL_DEFECTIVE = 10

EVAL_EVERY = 50

DEVICE = torch.device("cpu")

BEST_CHECKPOINT = Path(
    "outputs/checkpoints/short_balanced_best.pt"
)


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


def evaluate(model, loader):
    model.eval()

    total_dice = 0.0
    total_samples = 0

    healthy_total = 0
    healthy_false_positives = 0

    defective_total = 0
    defective_detected = 0

    max_probability = 0.0

    with torch.no_grad():

        for images, masks, labels in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            max_probability = max(
                max_probability,
                probabilities.max().item(),
            )

            for i in range(images.size(0)):

                image_dice = dice_score(
                    logits[i:i + 1],
                    masks[i:i + 1],
                )

                total_dice += image_dice
                total_samples += 1

                predicted_pixels = (
                    predictions[i].sum().item()
                )

                ground_truth_pixels = (
                    masks[i].sum().item()
                )

                if ground_truth_pixels == 0:

                    healthy_total += 1

                    if predicted_pixels > 0:
                        healthy_false_positives += 1

                else:

                    defective_total += 1

                    if predicted_pixels > 0:
                        defective_detected += 1

    model.train()

    mean_dice = (
        total_dice / total_samples
        if total_samples > 0
        else 0.0
    )

    detection_rate = (
        defective_detected / defective_total
        if defective_total > 0
        else 0.0
    )

    false_positive_rate = (
        healthy_false_positives / healthy_total
        if healthy_total > 0
        else 0.0
    )

    return (
        mean_dice,
        detection_rate,
        false_positive_rate,
        max_probability,
    )


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]

        image, mask = self.base_dataset[real_index]

        return (
            image,
            mask,
            int(
                mask.sum().item() > 0
            ),
        )


def main():

    print("L-01 SHORT BALANCED TRAINING + VALIDATION")
    print("==========================================")

    print(f"Device:          {DEVICE}")
    print(f"Target size:     {TARGET_SIZE}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Iterations:      {NUM_ITERATIONS}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print()

    train_dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    train_data = pd.read_csv(TRAIN_CSV)

    healthy_count = int(
        (train_data["has_defect"] == 0).sum()
    )

    defective_count = int(
        (train_data["has_defect"] == 1).sum()
    )

    weights = []

    for label in train_data["has_defect"]:

        if label == 0:
            weights.append(
                1.0 / healthy_count
            )
        else:
            weights.append(
                1.0 / defective_count
            )

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=NUM_ITERATIONS * BATCH_SIZE,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    # --------------------------------------------------
    # Fixed validation subset
    # --------------------------------------------------

    val_dataset = KolektorDataset(
        VAL_CSV,
        target_size=TARGET_SIZE,
    )

    val_data = pd.read_csv(VAL_CSV)

    healthy_indices = (
        val_data.index[
            val_data["has_defect"] == 0
        ].tolist()
    )

    defective_indices = (
        val_data.index[
            val_data["has_defect"] == 1
        ].tolist()
    )

    validation_indices = (
        healthy_indices[:VAL_HEALTHY]
        + defective_indices[:VAL_DEFECTIVE]
    )

    fixed_val_dataset = EvalDataset(
        val_dataset,
        validation_indices,
    )

    val_loader = DataLoader(
        fixed_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("FIXED VALIDATION SET")
    print("--------------------")
    print(f"Healthy samples:   {VAL_HEALTHY}")
    print(f"Defective samples: {VAL_DEFECTIVE}")
    print(f"Total samples:     {len(fixed_val_dataset)}")
    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UNet().to(DEVICE)

    criterion = WeightedBCEDiceLoss(
    pos_weight=20.0,
    bce_weight=0.5,
    dice_weight=0.5,
)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_dice = -1.0
    best_iteration = 0

    model.train()

    print("TRAINING + VALIDATION")
    print("---------------------")

    for iteration, (images, masks) in enumerate(
        train_loader,
        start=1,
    ):

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

        if iteration % EVAL_EVERY == 0:

            train_dice = dice_score(
                logits,
                masks,
            )

            (
                val_dice,
                detection_rate,
                false_positive_rate,
                max_probability,
            ) = evaluate(
                model,
                val_loader,
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_iteration = iteration

                BEST_CHECKPOINT.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                torch.save(
                    {
                        "iteration": iteration,
                        "val_dice": val_dice,
                        "detection_rate": detection_rate,
                        "false_positive_rate": false_positive_rate,
                        "model_state_dict": model.state_dict(),
                    },
                    BEST_CHECKPOINT,
                )

                print(
                    f"  -> New best checkpoint saved "
                    f"(iteration={iteration}, "
                    f"val_dice={val_dice:.6f})"
                )



            print(
                f"Iteration {iteration:3d} | "
                f"loss={loss.item():.6f} | "
                f"train_dice={train_dice:.6f} | "
                f"val_dice={val_dice:.6f} | "
                f"val_detection={detection_rate:.4f} | "
                f"val_FP={false_positive_rate:.4f} | "
                f"val_max_prob={max_probability:.6f}"
            )

    print()
    print("SHORT BALANCED VALIDATION COMPLETED")
    print("------------------------------------")
    print(f"Best iteration: {best_iteration}")
    print(f"Best val Dice:  {best_val_dice:.6f}")
    print(f"Checkpoint:     {BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()