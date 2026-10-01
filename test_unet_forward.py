import torch

from unet import UNet


def test_unet_forward():
    model = UNet()
    model.eval()

    batch_size = 2

    x = torch.randn(
        batch_size,
        1,
        384,
        384,
    )

    with torch.no_grad():
        output = model(x)

    print()
    print("U-NET FORWARD PASS AUDIT")
    print("------------------------")
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(
        f"Output range: "
        f"{output.min().item():.4f} - "
        f"{output.max().item():.4f}"
    )

    assert output.shape == (
        batch_size,
        1,
        384,
        384,
    )

    assert output.dtype == torch.float32


if __name__ == "__main__":
    test_unet_forward()