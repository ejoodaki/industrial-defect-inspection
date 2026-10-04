import torch
from torch.utils.data import DataLoader, Subset

from dataset import KolektorDataset


def test_dataloader_defective_batch_audit():
    dataset = KolektorDataset(
        "splits/train.csv",
        target_size=384,
    )

    defective_indices = [
        index
        for index in range(len(dataset))
        if dataset.data.iloc[index]["has_defect"] == 1
    ]

    assert len(defective_indices) > 0

    selected_indices = defective_indices[:4]

    subset = Subset(dataset, selected_indices)

    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    defect_pixels_per_sample = masks.sum(
        dim=(1, 2, 3)
    )

    print()
    print("DEFECTIVE DATALOADER BATCH AUDIT")
    print("--------------------------------")
    print(f"Defective dataset samples: {len(defective_indices)}")
    print(f"Selected indices: {selected_indices}")
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Mask batch shape: {tuple(masks.shape)}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")
    print(f"Image range: {images.min().item():.4f} - {images.max().item():.4f}")
    print(f"Mask unique values: {torch.unique(masks).tolist()}")
    print(
        "Defect pixels per sample: "
        f"{defect_pixels_per_sample.tolist()}"
    )

    assert images.shape == (4, 1, 384, 384)
    assert masks.shape == (4, 1, 384, 384)

    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32

    assert 0.0 <= images.min() <= 1.0
    assert images.max() <= 1.0

    assert set(
        torch.unique(masks).tolist()
    ).issubset({0.0, 1.0})

    assert torch.all(
        defect_pixels_per_sample > 0
    )