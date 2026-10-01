from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PATCH_SIZE = 192


class PatchClassifierDataset(Dataset):
    def __init__(self, csv_paths):
        self.samples = []

        for csv_path in csv_paths:
            csv_path = Path(csv_path)

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}"
                )

            dataframe = pd.read_csv(csv_path)

            for _, row in dataframe.iterrows():
                self.samples.append(row.to_dict())

        if not self.samples:
            raise ValueError("Dataset is empty.")

    @staticmethod
    def load_grayscale(path):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {path}"
            )

        return image

    @staticmethod
    def get_defect_bbox(mask):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )

    @staticmethod
    def crop_by_bbox(image, x1, y1, x2, y2):
        patch = image[y1:y2, x1:x2]

        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Invalid patch shape: {patch.shape}"
            )

        return patch

    def make_positive_patch(self, image, mask):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            raise ValueError(
                "Positive sample has no defect."
            )

        # Select a real defect pixel.
        # Using the median defect pixel keeps the crop
        # deterministic while guaranteeing that the
        # crop is centered around actual defect content.
        center_x = float(np.median(xs))
        center_y = float(np.median(ys))

        height, width = image.shape
        half = PATCH_SIZE // 2

        patch_x1 = int(round(center_x - half))
        patch_y1 = int(round(center_y - half))

        patch_x1 = max(
            0,
            min(patch_x1, width - PATCH_SIZE),
        )

        patch_y1 = max(
            0,
            min(patch_y1, height - PATCH_SIZE),
        )

        patch_x2 = patch_x1 + PATCH_SIZE
        patch_y2 = patch_y1 + PATCH_SIZE

        patch = image[
            patch_y1:patch_y2,
            patch_x1:patch_x2,
        ]

        mask_patch = mask[
            patch_y1:patch_y2,
            patch_x1:patch_x2,
        ]

        if patch.shape != (
            PATCH_SIZE,
            PATCH_SIZE,
        ):
            raise ValueError(
                f"Invalid positive patch shape: "
                f"{patch.shape}"
            )

        if not np.any(mask_patch > 0):
            raise ValueError(
                "Positive patch contains no defect pixels."
            )

        return patch

    def make_coordinate_patch(self, image, row):
        required = ["x1", "y1", "x2", "y2"]

        for key in required:
            if key not in row:
                raise ValueError(
                    f"Missing coordinate '{key}' "
                    f"for {row.get('patch_type')}"
                )

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        return self.crop_by_bbox(
            image,
            x1,
            y1,
            x2,
            y2,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]

        image = self.load_grayscale(
            row["image_path"]
        )

        label = int(row["label"])
        patch_type = row["patch_type"]

        if patch_type == "positive":
            mask = self.load_grayscale(
                row["mask_path"]
            )

            patch = self.make_positive_patch(
                image,
                mask,
            )

        elif patch_type in {
            "healthy_negative",
            "hard_negative",
        }:
            patch = self.make_coordinate_patch(
                image,
                row,
            )

        else:
            raise ValueError(
                f"Unknown patch type: {patch_type}"
            )

        patch = patch.astype(
            np.float32
        ) / 255.0

        patch = torch.from_numpy(
            patch
        ).unsqueeze(0)

        label = torch.tensor(
            [label],
            dtype=torch.float32,
        )

        return patch, label