from pathlib import Path

import pandas as pd
import torch

from dataset import KolektorDataset


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
VAL_CSV = ROOT / "splits" / "validation.csv"


def audit_tensor(image, mask, name):
    print(f"--- {name} ---")

    print(f"Image shape           : {tuple(image.shape)}")
    print(f"Mask shape            : {tuple(mask.shape)}")

    print(f"Image dtype           : {image.dtype}")
    print(f"Mask dtype            : {mask.dtype}")

    print(
        f"Image range           : "
        f"{image.min().item():.4f} - {image.max().item():.4f}"
    )

    print(
        f"Mask unique values    : "
        f"{torch.unique(mask).tolist()}"
    )

    defect_pixels = int((mask == 1.0).sum().item())

    print(f"Defect pixels         : {defect_pixels}")

    assert image.ndim == 3
    assert mask.ndim == 3

    assert image.shape[0] == 1
    assert mask.shape[0] == 1

    assert image.shape[1:] == mask.shape[1:]

    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32

    assert 0.0 <= image.min() <= image.max() <= 1.0

    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

    print("PASSED")


def audit_split(csv_path, name):
    dataset = KolektorDataset(csv_path)
    df = pd.read_csv(csv_path)

    print(f"=== {name.upper()} DATASET ===")
    print(f"Samples              : {len(dataset)}")
    print()

    # Healthy sample
    healthy_index = df.index[df["has_defect"] == 0][0]

    healthy_image, healthy_mask = dataset[healthy_index]

    audit_tensor(
        healthy_image,
        healthy_mask,
        f"Healthy sample index {healthy_index}",
    )

    assert int((healthy_mask == 1.0).sum().item()) == 0

    print("Healthy mask check   : PASSED")
    print()

    # Defective sample
    defective_index = df.index[df["has_defect"] == 1][0]

    defective_image, defective_mask = dataset[defective_index]

    audit_tensor(
        defective_image,
        defective_mask,
        f"Defective sample index {defective_index}",
    )

    defective_pixels = int(
        (defective_mask == 1.0).sum().item()
    )

    assert defective_pixels > 0

    print("Defective mask check : PASSED")
    print()


if __name__ == "__main__":
    audit_split(TRAIN_CSV, "train")

    print()

    audit_split(VAL_CSV, "validation")

    print()
    print("DATASET TENSOR AUDIT COMPLETED SUCCESSFULLY")