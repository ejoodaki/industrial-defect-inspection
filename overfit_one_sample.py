import torch
import torch.nn as nn

from dataset import KolektorDataset
from losses import BCEDiceLoss
from unet import UNet


TRAIN_CSV = "splits/train.csv"

SAMPLE_INDEX = 21

TARGET_SIZE = 384

ITERATIONS = 300

LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


def dice_score(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    intersection = (
        predictions * targets
    ).sum()

    denominator = (
        predictions.sum()
        + targets.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        (2.0 * intersection) /
        denominator
    )


def main():
    print("L-01 SINGLE-SAMPLE OVERFIT TEST")
    print("================================")

    print(f"Device:     {DEVICE}")
    print(f"Sample:     {SAMPLE_INDEX}")
    print(f"Iterations: {ITERATIONS}")
    print(f"LR:         {LEARNING_RATE}")
    print()

    dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    image, mask = dataset[SAMPLE_INDEX]

    image = image.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)

    print(f"Image shape: {tuple(image.shape)}")
    print(f"Mask shape:  {tuple(mask.shape)}")
    print(
        f"Ground-truth defect pixels: "
        f"{int(mask.sum().item())}"
    )
    print()

    model = UNet(
        in_channels=1,
        out_channels=1,
    )

    model.to(DEVICE)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    best_dice = 0.0

    for iteration in range(1, ITERATIONS + 1):

        optimizer.zero_grad()

        logits = model(image)

        loss = criterion(
            logits,
            mask,
        )

        loss.backward()

        optimizer.step()

        if (
            iteration == 1
            or iteration % 25 == 0
            or iteration == ITERATIONS
        ):
            model.eval()

            with torch.no_grad():
                current_logits = model(image)

                probabilities = torch.sigmoid(
                    current_logits
                )

                max_probability = float(
                    probabilities.max().item()
                )

                current_dice = dice_score(
                    current_logits,
                    mask,
                )

            model.train()

            best_dice = max(
                best_dice,
                current_dice,
            )

            print(
                f"Iteration {iteration:3d} | "
                f"loss={loss.item():.6f} | "
                f"dice={current_dice:.6f} | "
                f"max_prob={max_probability:.6f}"
            )

    print()
    print("FINAL RESULT")
    print("------------")
    print(f"Best Dice: {best_dice:.6f}")

    if best_dice >= 0.90:
        print(
            "STATUS: PASS - "
            "model can overfit one sample."
        )
    else:
        print(
            "STATUS: FAIL - "
            "model could not overfit one sample."
        )

    print()
    print("SINGLE-SAMPLE OVERFIT TEST COMPLETED")


if __name__ == "__main__":
    main()