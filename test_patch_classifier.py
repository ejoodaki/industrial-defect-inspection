import torch

from patch_classifier import PatchClassifier


def main():
    model = PatchClassifier()

    x = torch.randn(2, 1, 192, 192)

    with torch.no_grad():
        logits = model(x)

    print("PATCH CLASSIFIER AUDIT")
    print("----------------------")
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Output dtype: {logits.dtype}")
    print(f"Output range: {logits.min().item():.6f} -> {logits.max().item():.6f}")

    assert logits.shape == (2, 1)
    assert logits.dtype == torch.float32

    print()
    print("PATCH CLASSIFIER FORWARD PASSED")


if __name__ == "__main__":
    main()