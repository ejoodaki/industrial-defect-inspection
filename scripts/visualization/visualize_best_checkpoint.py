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
    "outputs/checkpoint_visualization"
)

TARGET_SIZE = 384
THRESHOLD = 0.5

DEVICE = torch.device("cpu")


def main():

    print("L-01 BEST CHECKPOINT VISUALIZATION")
    print("==================================")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device:     {DEVICE}")
    print(f"Threshold:  {THRESHOLD}")
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

    # Small targeted audit only.
    selected_indices = (
        defective_indices[:4]
        + healthy_indices[:2]
    )

    print("SELECTED SAMPLES")
    print("-----------------")
    print(f"Defective: {defective_indices[:4]}")
    print(f"Healthy:   {healthy_indices[:2]}")
    print()

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

    print("CHECKPOINT INFO")
    print("---------------")
    print(
        f"Iteration:          "
        f"{checkpoint['iteration']}"
    )
    print(
        f"Validation Dice:    "
        f"{checkpoint['val_dice']:.6f}"
    )
    print(
        f"Detection rate:     "
        f"{checkpoint['detection_rate']:.4f}"
    )
    print(
        f"False positive rate:"
        f" {checkpoint['false_positive_rate']:.4f}"
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

            ground_truth = (
                mask[0].numpy()
            )

            image_np = (
                image[0].numpy()
            )

            prediction = (
                probability >= THRESHOLD
            ).astype(np.float32)

            gt_pixels = int(
                ground_truth.sum()
            )

            predicted_pixels = int(
                prediction.sum()
            )

            max_probability = float(
                probability.max()
            )

            mean_probability = float(
                probability.mean()
            )

            print(
                f"Sample {index:4d} | "
                f"GT pixels={gt_pixels:5d} | "
                f"Pred pixels={predicted_pixels:5d} | "
                f"max_prob={max_probability:.6f} | "
                f"mean_prob={mean_probability:.8f}"
            )

            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(16, 5),
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

            axes[2].imshow(
                prediction,
                cmap="gray",
            )
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            axes[3].imshow(
                image_np,
                cmap="gray",
            )
            axes[3].imshow(
                prediction,
                cmap="Reds",
                alpha=0.5,
            )
            axes[3].set_title("Prediction Overlay")
            axes[3].axis("off")

            fig.suptitle(
                f"Sample {index} | "
                f"GT={gt_pixels} px | "
                f"Pred={predicted_pixels} px"
            )

            output_path = (
                OUTPUT_DIR
                / f"sample_{index:04d}.png"
            )

            plt.tight_layout()
            plt.savefig(
                output_path,
                dpi=150,
            )
            plt.close(fig)

    print()
    print("VISUALIZATION COMPLETED")
    print("-----------------------")
    print(
        f"Saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()