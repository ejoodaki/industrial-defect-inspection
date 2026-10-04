import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from unet import UNet


def main():
    dataset = KolektorDataset(
        "splits/train.csv",
        target_size=384,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    model = UNet()
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    print()
    print("REAL DATA U-NET FORWARD AUDIT")
    print("-----------------------------")
    print(f"Images shape:  {tuple(images.shape)}")
    print(f"Masks shape:   {tuple(masks.shape)}")
    print(f"Output shape:  {tuple(outputs.shape)}")
    print(f"Image dtype:   {images.dtype}")
    print(f"Mask dtype:    {masks.dtype}")
    print(f"Output dtype:  {outputs.dtype}")
    print(
        f"Output range: "
        f"{outputs.min().item():.4f} - "
        f"{outputs.max().item():.4f}"
    )

    assert images.shape == (2, 1, 384, 384)
    assert masks.shape == (2, 1, 384, 384)
    assert outputs.shape == (2, 1, 384, 384)

    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32
    assert outputs.dtype == torch.float32


if __name__ == "__main__":
    main()