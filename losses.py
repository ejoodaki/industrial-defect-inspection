import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Expects:
        logits: (B, 1, H, W)
        targets: (B, 1, H, W)

    The sigmoid is applied internally.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.size(0),
            -1,
        )

        targets = targets.reshape(
            targets.size(0),
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogits + Dice loss.

    total_loss = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss

class WeightedBCEDiceLoss(nn.Module):
    """
    Combined weighted BCEWithLogits + Dice loss.

    Positive pixels receive a higher BCE weight through pos_weight.
    """

    def __init__(
        self,
        pos_weight=10.0,
        bce_weight=0.5,
        dice_weight=0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )

        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss