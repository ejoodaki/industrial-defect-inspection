from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from transforms import PairedResizePad
from unet import UNet
from losses import WeightedBCEDiceLoss


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
HARD_NEGATIVE_CSV = (
    ROOT / "outputs" / "hard_negative_audit"
    / "hard_negative_candidates.csv"
)

PATCH_SIZE = 192
BATCH_SIZE = 2
ITERATIONS = 200
LEARNING_RATE = 0.001

SEED = 42

POSITIVE_PROBABILITY = 0.5

DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Failed to read: {path}")

    return image


def crop_patch(image, x, y):
    return image[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE,
    ]


def get_positive_crop(image, mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError("Positive sample has no defect.")

    center_x = int((xs.min() + xs.max()) / 2)
    center_y = int((ys.min() + ys.max()) / 2)

    x = center_x - PATCH_SIZE // 2
    y = center_y - PATCH_SIZE // 2

    h, w = image.shape

    x = max(0, min(x, w - PATCH_SIZE))
    y = max(0, min(y, h - PATCH_SIZE))

    image_crop = crop_patch(image, x, y)
    mask_crop = crop_patch(mask, x, y)

    return image_crop, mask_crop


class HardNegativePatchDataset(Dataset):

    def __init__(self, train_df, hard_negative_df):
        self.train_df = train_df.reset_index(drop=True)
        self.hard_negative_df = hard_negative_df.reset_index(drop=True)

        self.positive_rows = self.train_df[
            self.train_df["has_defect"] == 1
        ].reset_index(drop=True)

    def __len__(self):
        return 1000000

    def __getitem__(self, index):

        if random.random() < POSITIVE_PROBABILITY:

            row = self.positive_rows.sample(
                n=1
            ).iloc[0]

            image_path = ROOT / row["image_path"]
            mask_path = ROOT / row["mask_path"]

            image = load_gray(image_path)
            mask = load_gray(mask_path)

            image, mask = get_positive_crop(
                image,
                mask,
            )

            label = "positive"

        else:

            row = self.hard_negative_df.sample(
                n=1
            ).iloc[0]

            image_path = Path(row["image_path"])

            image = load_gray(image_path)

            x = int(row["x"])
            y = int(row["y"])

            image = crop_patch(
                image,
                x,
                y,
            )

            mask = np.zeros_like(image)

            label = "hard_negative"

        transform = PairedResizePad(
            target_size=PATCH_SIZE
        )

        image, mask = transform(
            image,
            mask,
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask, label


def dice_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        predictions.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        (2 * intersection + 1e-6)
        / (denominator + 1e-6)
    )

    return dice.mean().item()


def evaluate(model, positive_rows, hard_negative_df):

    model.eval()

    positive_dices = []
    positive_detected = 0

    negative_false_positive = 0

    transform = PairedResizePad(
        target_size=PATCH_SIZE
    )

    with torch.no_grad():

        # Positive validation
        for _, row in positive_rows.iterrows():

            image = load_gray(
                ROOT / row["image_path"]
            )

            mask = load_gray(
                ROOT / row["mask_path"]
            )

            image, mask = get_positive_crop(
                image,
                mask,
            )

            image, mask = transform(
                image,
                mask,
            )

            image = torch.from_numpy(
                image
            ).unsqueeze(0).unsqueeze(0)

            mask = torch.from_numpy(
                mask
            ).unsqueeze(0).unsqueeze(0)

            logits = model(image)

            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities >= 0.5
            ).float()

            if prediction.sum().item() > 0:
                positive_detected += 1

            intersection = (
                prediction * mask
            ).sum()

            denominator = (
                prediction.sum()
                + mask.sum()
            )

            dice = (
                (2 * intersection + 1e-6)
                / (denominator + 1e-6)
            )

            positive_dices.append(
                dice.item()
            )

        # Hard-negative validation
        negative_subset = hard_negative_df.sample(
            n=min(20, len(hard_negative_df)),
            random_state=SEED,
        )

        for _, row in negative_subset.iterrows():

            image = load_gray(
                Path(row["image_path"])
            )

            x = int(row["x"])
            y = int(row["y"])

            image = crop_patch(
                image,
                x,
                y,
            )

            mask = np.zeros_like(image)

            image, mask = transform(
                image,
                mask,
            )

            image = torch.from_numpy(
                image
            ).unsqueeze(0).unsqueeze(0)

            logits = model(image)

            prediction = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            if prediction.sum().item() > 0:
                negative_false_positive += 1

    model.train()

    detection_rate = (
        positive_detected / len(positive_rows)
    )

    false_positive_rate = (
        negative_false_positive
        / min(20, len(hard_negative_df))
    )

    return (
        float(np.mean(positive_dices)),
        detection_rate,
        false_positive_rate,
    )


def main():

    set_seed(SEED)

    print("HARD-NEGATIVE PATCH TRAINING POC")
    print("================================")
    print(f"Device:          {DEVICE}")
    print(f"Patch size:      {PATCH_SIZE}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Iterations:      {ITERATIONS}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print()

    train_df = pd.read_csv(TRAIN_CSV)

    hard_negative_df = pd.read_csv(
        HARD_NEGATIVE_CSV
    )

    positive_rows = train_df[
        train_df["has_defect"] == 1
    ].sample(
        n=10,
        random_state=SEED,
    ).reset_index(drop=True)

    dataset = HardNegativePatchDataset(
        train_df,
        hard_negative_df,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    iterator = iter(loader)

    model = UNet(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)

    criterion = WeightedBCEDiceLoss(
        pos_weight=10.0,
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_dice = -1.0

    checkpoint_dir = (
        ROOT / "outputs" / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / "hard_negative_patch_poc_best.pt"
    )

    for iteration in range(
        1,
        ITERATIONS + 1,
    ):

        try:
            images, masks, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, masks, labels = next(iterator)

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

        if iteration % 50 == 0:

            train_dice = dice_score(
                logits,
                masks,
            )

            val_dice, detection, false_positive = evaluate(
                model,
                positive_rows,
                hard_negative_df,
            )

            print(
                f"Iteration {iteration:3d} | "
                f"loss={loss.item():.6f} | "
                f"train_dice={train_dice:.6f} | "
                f"val_positive_dice={val_dice:.6f} | "
                f"val_detection={detection:.4f} | "
                f"hard_negative_FP={false_positive:.4f}"
            )

            if val_dice > best_dice:

                best_dice = val_dice

                torch.save(
                    {
                        "iteration": iteration,
                        "val_positive_dice": val_dice,
                        "detection_rate": detection,
                        "hard_negative_false_positive_rate": (
                            false_positive
                        ),
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )

                print(
                    "  -> New best checkpoint saved"
                )

    print()
    print("HARD-NEGATIVE PATCH POC COMPLETED")
    print("----------------------------------")
    print(
        f"Best positive Dice: {best_dice:.6f}"
    )
    print(
        f"Checkpoint: {checkpoint_path}"
    )


if __name__ == "__main__":
    main()