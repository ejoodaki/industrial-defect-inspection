from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from dataset import KolektorDataset


OUTPUT_DIR = Path("outputs") / "transform_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    dataset = KolektorDataset(
        "splits/train.csv",
        target_size=384,
    )

    defective_indices = [
        index
        for index in range(len(dataset))
        if dataset.data.iloc[index]["has_defect"] == 1
    ]

    selected_indices = defective_indices[:4]

    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    for batch_index in range(4):
        image = images[batch_index, 0].numpy()
        mask = masks[batch_index, 0].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Transformed Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Transformed Mask")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(
            mask,
            alpha=0.35,
            cmap="Reds",
        )
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        sample_id = Path(
            dataset.data.iloc[selected_indices[batch_index]]["image_path"]
        ).stem

        output_path = (
            OUTPUT_DIR
            / f"{sample_id}_transform_audit.png"
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()