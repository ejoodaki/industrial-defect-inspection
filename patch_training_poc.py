import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transforms import PairedResizePad
from unet import UNet
from losses import WeightedBCEDiceLoss


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
VAL_CSV = ROOT / "splits" / "validation.csv"

PATCH_SIZE = 192
BATCH_SIZE = 2
ITERATIONS = 200
LEARNING_RATE = 0.001

SEED = 42

DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_pair(row):
    image_path = ROOT / row["image_path"]
    mask_path = ROOT / row["mask_path"]

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    if image.shape != mask.shape:
        raise RuntimeError(
            f"Image/mask mismatch: "
            f"{image.shape} vs {mask.shape}"
        )

    mask = (mask > 0).astype(np.uint8) * 255

    return image, mask


def get_bbox_center(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    cx = int((xs.min() + xs.max()) // 2)
    cy = int((ys.min() + ys.max()) // 2)

    return cx, cy


def extract_centered_patch(image, mask, cx, cy, patch_size):
    h, w = image.shape

    half = patch_size // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    if x0 < 0:
        x1 -= x0
        x0 = 0

    if y0 < 0:
        y1 -= y0
        y0 = 0

    if x1 > w:
        x0 -= x1 - w
        x1 = w

    if y1 > h:
        y0 -= y1 - h
        y1 = h

    x0 = max(0, x0)
    y0 = max(0, y0)

    x1 = min(w, x1)
    y1 = min(h, y1)

    image_patch = image[y0:y1, x0:x1]
    mask_patch = mask[y0:y1, x0:x1]

    if image_patch.shape != (patch_size, patch_size):
        padded_image = np.zeros(
            (patch_size, patch_size),
            dtype=np.uint8,
        )

        padded_mask = np.zeros(
            (patch_size, patch_size),
            dtype=np.uint8,
        )

        ph, pw = image_patch.shape

        padded_image[:ph, :pw] = image_patch
        padded_mask[:ph, :pw] = mask_patch

        image_patch = padded_image
        mask_patch = padded_mask

    return image_patch, mask_patch


def extract_random_patch(image, mask, patch_size):
    h, w = image.shape

    if h < patch_size or w < patch_size:
        raise RuntimeError(
            f"Image too small for patch: "
            f"{image.shape}"
        )

    max_y = h - patch_size
    max_x = w - patch_size

    y0 = random.randint(0, max_y)
    x0 = random.randint(0, max_x)

    image_patch = image[
        y0:y0 + patch_size,
        x0:x0 + patch_size,
    ]

    mask_patch = mask[
        y0:y0 + patch_size,
        x0:x0 + patch_size,
    ]

    return image_patch, mask_patch


class PatchDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

        self.positive_rows = [
            row
            for row in rows
            if row["has_defect"].strip() == "1"
        ]

        self.negative_rows = [
            row
            for row in rows
            if row["has_defect"].strip() == "0"
        ]

        self.transform = PairedResizePad(
            target_size=PATCH_SIZE
        )

    def __len__(self):
        return max(
            len(self.positive_rows),
            len(self.negative_rows),
        ) * 2

    def __getitem__(self, index):
        is_positive = index % 2 == 0

        if is_positive:
            row = random.choice(self.positive_rows)

            image, mask = read_pair(row)

            center = get_bbox_center(mask)

            if center is None:
                raise RuntimeError(
                    f"Expected defect but mask is empty: "
                    f"{row['image_path']}"
                )

            cx, cy = center

            image, mask = extract_centered_patch(
                image,
                mask,
                cx,
                cy,
                PATCH_SIZE,
            )

        else:
            row = random.choice(self.negative_rows)

            image, mask = read_pair(row)

            image, mask = extract_random_patch(
                image,
                mask,
                PATCH_SIZE,
            )

        image, mask = self.transform(
            image,
            mask,
        )

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image.float(), mask.float()


def dice_from_logits(logits, targets):
    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

    intersection = (
        predictions * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        predictions.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = torch.where(
        denominator > 0,
        (2.0 * intersection + 1e-6)
        / (denominator + 1e-6),
        torch.ones_like(denominator),
    )

    return dice.mean().item()


@torch.no_grad()
def evaluate_patch_model(
    model,
    positive_rows,
    negative_rows,
):
    model.eval()

    positive_dice = []
    positive_detected = 0

    negative_false_positive = 0

    transform = PairedResizePad(
        target_size=PATCH_SIZE
    )

    # Positive diagnostic patches.
    for row in positive_rows[:10]:
        image, mask = read_pair(row)

        center = get_bbox_center(mask)

        if center is None:
            continue

        cx, cy = center

        image, mask = extract_centered_patch(
            image,
            mask,
            cx,
            cy,
            PATCH_SIZE,
        )

        image, mask = transform(
            image,
            mask,
        )

        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        logits = model(image)

        dice = dice_from_logits(
            logits,
            mask,
        )

        positive_dice.append(dice)

        prediction = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        if prediction.sum().item() > 0:
            positive_detected += 1

    # Healthy diagnostic patches.
    for row in negative_rows[:10]:
        image, mask = read_pair(row)

        image, mask = extract_random_patch(
            image,
            mask,
            PATCH_SIZE,
        )

        image, mask = transform(
            image,
            mask,
        )

        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        logits = model(image)

        prediction = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        if prediction.sum().item() > 0:
            negative_false_positive += 1

    mean_dice = (
        float(np.mean(positive_dice))
        if positive_dice
        else 0.0
    )

    detection_rate = (
        positive_detected / len(positive_dice)
        if positive_dice
        else 0.0
    )

    false_positive_rate = (
        negative_false_positive / 10
    )

    return (
        mean_dice,
        detection_rate,
        false_positive_rate,
    )


def main():
    set_seed(SEED)

    train_rows = read_csv(TRAIN_CSV)
    val_rows = read_csv(VAL_CSV)

    train_dataset = PatchDataset(train_rows)

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    loader_iter = iter(loader)

    positive_val = [
        row
        for row in val_rows
        if row["has_defect"].strip() == "1"
    ]

    negative_val = [
        row
        for row in val_rows
        if row["has_defect"].strip() == "0"
    ]

    model = UNet(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)

    criterion = WeightedBCEDiceLoss(
        pos_weight=10.0,
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("L-01 PATCH TRAINING PROOF-OF-CONCEPT")
    print("======================================")
    print(f"Device:          {DEVICE}")
    print(f"Patch size:      {PATCH_SIZE}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Iterations:      {ITERATIONS}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print()
    print("TRAINING DATA")
    print("-------------")
    print(f"Total rows:      {len(train_rows)}")
    print(
        f"Positive rows:   "
        f"{len(train_dataset.positive_rows)}"
    )
    print(
        f"Negative rows:   "
        f"{len(train_dataset.negative_rows)}"
    )
    print()
    print("VALIDATION")
    print("----------")
    print("Positive diagnostic patches: 10")
    print("Healthy diagnostic patches:  10")
    print()

    best_dice = -1.0

    start_time = time.time()

    for iteration in range(1, ITERATIONS + 1):

        try:
            images, masks = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, masks = next(loader_iter)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        loss.backward()

        optimizer.step()

        if iteration % 50 == 0:

            train_dice = dice_from_logits(
                logits.detach(),
                masks,
            )

            val_dice, detection_rate, fp_rate = (
                evaluate_patch_model(
                    model,
                    positive_val,
                    negative_val,
                )
            )

            print(
                f"Iteration {iteration:3d} | "
                f"loss={loss.item():.6f} | "
                f"train_dice={train_dice:.6f} | "
                f"val_positive_dice={val_dice:.6f} | "
                f"val_detection={detection_rate:.4f} | "
                f"healthy_FP={fp_rate:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice

                output_dir = (
                    ROOT
                    / "outputs"
                    / "checkpoints"
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                checkpoint_path = (
                    output_dir
                    / "patch_poc_best.pt"
                )

                torch.save(
                    {
                        "iteration": iteration,
                        "val_positive_dice": val_dice,
                        "val_detection_rate": detection_rate,
                        "healthy_false_positive_rate": fp_rate,
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )

                print(
                    "  -> New best checkpoint saved"
                )

    elapsed = time.time() - start_time

    print()
    print("PATCH TRAINING POC COMPLETED")
    print("----------------------------")
    print(f"Best positive Dice: {best_dice:.6f}")
    print(f"Elapsed time:       {elapsed:.1f}s")
    print(
        "Checkpoint: "
        "outputs\\checkpoints\\patch_poc_best.pt"
    )


if __name__ == "__main__":
    main()