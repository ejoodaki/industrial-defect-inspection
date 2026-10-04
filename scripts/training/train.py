import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from losses import BCEDiceLoss
from unet import UNet


DEVICE = torch.device("cpu")

TRAIN_CSV = "splits/train.csv"
VAL_CSV = "splits/validation.csv"

TARGET_SIZE = 384
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
EPOCHS = 3

CHECKPOINT_DIR = Path("outputs/checkpoints")
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    running_loss = 0.0

    for images, masks in loader:
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

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    print()
    print("L-01 BASELINE TRAINING")
    print("======================")
    print(f"Device:        {DEVICE}")
    print(f"Target size:   {TARGET_SIZE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs:        {EPOCHS}")

    print()
    print("Loading datasets...")

    train_dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    val_dataset = KolektorDataset(
        VAL_CSV,
        target_size=TARGET_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    print()
    print("Building model...")

    model = UNet().to(DEVICE)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Model ready.")
    print()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"time={epoch_time:.1f}s"
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"baseline_epoch_{epoch}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = (
                CHECKPOINT_DIR
                / "baseline_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_path,
            )

            print(
                f"Best checkpoint updated: {best_path}"
            )

    print()
    print("BASELINE TRAINING COMPLETED")
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


if __name__ == "__main__":
    main()