from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from unet import UNet


CHECKPOINT = Path("outputs/checkpoints/baseline_best.pt")
TRAIN_CSV = Path("splits/train.csv")

BATCH_SIZE = 2
TARGET_SIZE = 384

DEVICE = torch.device("cpu")


def main():
    print("L-01 TRAIN RAW PREDICTION AUDIT")
    print("===============================")

    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print()

    dataset = KolektorDataset(
        TRAIN_CSV,
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    healthy_max_probs = []
    defective_max_probs = []

    healthy_mean_probs = []
    defective_mean_probs = []

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

                is_defective = bool(
                    (mask > 0.5).any()
                )

                if is_defective:
                    defective_max_probs.append(max_prob)
                    defective_mean_probs.append(mean_prob)
                else:
                    healthy_max_probs.append(max_prob)
                    healthy_mean_probs.append(mean_prob)

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

    print("TRAIN RAW PREDICTION AUDIT COMPLETED")


if __name__ == "__main__":
    main()