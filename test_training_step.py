import torch
from torch.utils.data import DataLoader

from dataset import KolektorDataset
from losses import BCEDiceLoss
from unet import UNet


def main():
    print()
    print("TRAINING STEP AUDIT")
    print("-------------------")

    device = torch.device("cpu")

    dataset = KolektorDataset(
        "splits/train.csv",
        target_size=384,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    model = UNet().to(device)
    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    images, masks = next(iter(loader))

    images = images.to(device)
    masks = masks.to(device)

    print(f"Images shape: {tuple(images.shape)}")
    print(f"Masks shape:  {tuple(masks.shape)}")

    # Forward
    logits = model(images)

    print(f"Logits shape: {tuple(logits.shape)}")

    # Loss
    loss = criterion(logits, masks)

    print(f"Initial loss:  {loss.item():.6f}")

    assert logits.shape == masks.shape
    assert torch.isfinite(loss)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    gradient_found = False

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient_found = True
            break

    assert gradient_found

    print("Gradient check: PASSED")

    # Optimizer update
    optimizer.step()

    print("Optimizer step: PASSED")

    print()
    print("TRAINING STEP AUDIT PASSED")


if __name__ == "__main__":
    main()