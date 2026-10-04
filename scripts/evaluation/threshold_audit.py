from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import KolektorDataset
from unet import UNet


VAL_CSV = Path("splits/validation.csv")

CHECKPOINT = Path(
    "outputs/checkpoints/short_balanced_best.pt"
)

OUTPUT_DIR = Path(
    "outputs/threshold_audit"
)

TARGET_SIZE = 384

THRESHOLDS = [
    0.50,
    0.40,
    0.30,
    0.20,
]

DEVICE = torch.device("cpu")


def main():

    print("L-01 THRESHOLD AUDIT")
    print("====================")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device:     {DEVICE}")
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = KolektorDataset(
        VAL_CSV,
        target_size=TARGET_SIZE,
    )

    data = pd.read_csv(VAL_CSV)

    defective_indices = (
        data.index[
            data["has_defect"] == 1
        ].tolist()
    )

    healthy_indices = (
        data.index[
            data["has_defect"] == 0
        ].tolist()
    )

    selected_indices = (
        defective_indices[:4]
        + healthy_indices[:2]
    )

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
    # Prediction
    # --------------------------------------------------

    with torch.no_grad():

        for index in selected_indices:

            image, mask = dataset[index]

            input_tensor = (
                image
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(input_tensor)

            probability = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

            image_np = image[0].numpy()
            ground_truth = mask[0].numpy()

            gt_pixels = int(
                ground_truth.sum()
            )

            print()
            print(
                f"SAMPLE {index}"
            )
            print("-" * 30)
            print(
                f"Ground truth pixels: "
                f"{gt_pixels}"
            )
            print(
                f"Probability min:     "
                f"{probability.min():.6f}"
            )
            print(
                f"Probability max:     "
                f"{probability.max():.6f}"
            )
            print(
                f"Probability mean:    "
                f"{probability.mean():.6f}"
            )

            # --------------------------------------------------
            # Threshold statistics
            # --------------------------------------------------

            for threshold in THRESHOLDS:

                prediction = (
                    probability >= threshold
                )

                predicted_pixels = int(
                    prediction.sum()
                )

                print(
                    f"Threshold {threshold:.2f} "
                    f"-> predicted pixels: "
                    f"{predicted_pixels}"
                )

            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------

            fig, axes = plt.subplots(
                1,
                5,
                figsize=(20, 4),
            )

            axes[0].imshow(
                image_np,
                cmap="gray",
            )
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(
                ground_truth,
                cmap="gray",
            )
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            for axis, threshold in zip(
                axes[2:],
                THRESHOLDS[:3],
            ):

                prediction = (
                    probability >= threshold
                )

                axis.imshow(
                    image_np,
                    cmap="gray",
                )

                axis.imshow(
                    prediction,
                    cmap="Reds",
                    alpha=0.5,
                )

                axis.set_title(
                    f"Threshold {threshold:.2f}"
                )

                axis.axis("off")

            output_path = (
                OUTPUT_DIR
                / f"sample_{index:04d}_thresholds.png"
            )

            plt.tight_layout()

            plt.savefig(
                output_path,
                dpi=150,
            )

            plt.close(fig)

    print()
    print("THRESHOLD AUDIT COMPLETED")
    print("--------------------------")
    print(
        f"Saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()