from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import KolektorDataset
from unet import UNet


CHECKPOINT = Path("outputs/checkpoints/baseline_best.pt")
TRAIN_CSV = Path("splits/train.csv")

TARGET_SIZE = 384
SAMPLE_INDEX = 21

DEVICE = torch.device("cpu")

OUTPUT_DIR = Path("outputs/prediction_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("L-01 SINGLE-SAMPLE PREDICTION AUDIT")
    print("===================================")

    dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    image, mask = dataset[SAMPLE_INDEX]

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

    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(DEVICE)

        logits = model(input_tensor)
        probability = torch.sigmoid(logits)[0, 0]

    image_np = image[0].numpy()
    mask_np = mask[0].numpy()
    probability_np = probability.numpy()

    predicted_mask = (
        probability_np >= 0.01
    ).astype(np.float32)

    print(f"Sample index: {SAMPLE_INDEX}")
    print(f"Image shape:  {image_np.shape}")
    print(
        f"GT defect pixels: "
        f"{int(mask_np.sum())}"
    )

    print()
    print("RAW PREDICTION")
    print("--------------")
    print(
        f"Probability min:  "
        f"{probability_np.min():.8f}"
    )
    print(
        f"Probability max:  "
        f"{probability_np.max():.8f}"
    )
    print(
        f"Probability mean: "
        f"{probability_np.mean():.8f}"
    )

    print()
    print("THRESHOLD CHECK")
    print("---------------")

    for threshold in [
        0.001,
        0.005,
        0.01,
        0.02,
    ]:
        positive_pixels = int(
            (probability_np >= threshold).sum()
        )

        print(
            f"Threshold {threshold:.3f}: "
            f"{positive_pixels} positive pixels"
        )

    print()
    print("Saving visualization...")

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        image_np,
        cmap="gray",
    )
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(
        mask_np,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(
        probability_np,
        cmap="hot",
        vmin=0,
        vmax=max(
            0.02,
            probability_np.max(),
        ),
    )
    axes[2].set_title("Prediction Probability")
    axes[2].axis("off")

    axes[3].imshow(
        predicted_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Prediction @ 0.01")
    axes[3].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "train_sample_21_prediction_audit.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")
    print()
    print("SINGLE-SAMPLE PREDICTION AUDIT COMPLETED")


if __name__ == "__main__":
    main()