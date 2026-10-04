import torch

from losses import (
    DiceLoss,
    BCEDiceLoss,
    WeightedBCEDiceLoss,
)


def main():
    print()
    print("LOSS FUNCTION AUDIT")
    print("-------------------")

    dice_loss = DiceLoss()
    combined_loss = BCEDiceLoss()
    weighted_combined_loss = WeightedBCEDiceLoss(
     pos_weight=10.0,
    )

    # --------------------------------------------------
    # Case 1: Perfect prediction
    # --------------------------------------------------
    targets = torch.ones(1, 1, 4, 4)
    logits = torch.full(
        (1, 1, 4, 4),
        10.0,
    )

    perfect_dice = dice_loss(logits, targets)
    perfect_combined = combined_loss(
        logits,
        targets,
    )

    print()
    print("Case 1: Perfect positive prediction")
    print(f"Dice loss:     {perfect_dice.item():.6f}")
    print(f"Combined loss: {perfect_combined.item():.6f}")

    assert perfect_dice.item() < 0.01
    assert perfect_combined.item() < 0.01

    # --------------------------------------------------
    # Case 2: Completely wrong prediction
    # --------------------------------------------------
    logits = torch.full(
        (1, 1, 4, 4),
        -10.0,
    )

    bad_dice = dice_loss(logits, targets)
    bad_combined = combined_loss(
        logits,
        targets,
    )

    print()
    print("Case 2: Completely wrong prediction")
    print(f"Dice loss:     {bad_dice.item():.6f}")
    print(f"Combined loss: {bad_combined.item():.6f}")

    assert bad_dice.item() > perfect_dice.item()
    assert bad_combined.item() > perfect_combined.item()

    # --------------------------------------------------
    # Case 3: Healthy image / empty mask
    # --------------------------------------------------
    targets = torch.zeros(1, 1, 4, 4)
    logits = torch.full(
        (1, 1, 4, 4),
        -10.0,
    )

    healthy_dice = dice_loss(logits, targets)
    healthy_combined = combined_loss(
        logits,
        targets,
    )

    print()
    print("Case 3: Healthy image / empty mask")
    print(f"Dice loss:     {healthy_dice.item():.6f}")
    print(f"Combined loss: {healthy_combined.item():.6f}")

    assert torch.isfinite(healthy_dice)
    assert torch.isfinite(healthy_combined)

    # --------------------------------------------------
    # Case 4: Partial defect
    # --------------------------------------------------
    targets = torch.zeros(1, 1, 4, 4)
    targets[:, :, 1:3, 1:3] = 1.0

    logits = torch.full(
        (1, 1, 4, 4),
        -10.0,
    )

    logits[:, :, 1:3, 1:3] = 10.0

    partial_dice = dice_loss(logits, targets)
    partial_combined = combined_loss(
        logits,
        targets,
    )

    # --------------------------------------------------
    # Case 5: Weighted BCE + Dice
    # --------------------------------------------------
    weighted_loss = weighted_combined_loss(
        logits,
        targets,
    )

    print()
    print("Case 5: Weighted BCE + Dice")
    print(f"Weighted loss: {weighted_loss.item():.6f}")

    assert torch.isfinite(weighted_loss)
    assert weighted_loss.item() >= 0.0

    print()
    print("WEIGHTED LOSS AUDIT PASSED")

    print()
    print("Case 4: Partial defect prediction")
    print(f"Dice loss:     {partial_dice.item():.6f}")
    print(f"Combined loss: {partial_combined.item():.6f}")

    assert partial_dice.item() < 0.01
    assert partial_combined.item() < 0.01

    print()
    print("LOSS FUNCTION AUDIT PASSED")


if __name__ == "__main__":
    main()