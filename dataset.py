from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from transforms import PairedResizePad


class KolektorDataset(Dataset):
    def __init__(self, csv_path, target_size=384):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.transform = PairedResizePad(target_size=target_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = np.asarray(
            Image.open(row["image_path"]).convert("L"),
            dtype=np.uint8,
        )

        mask = np.asarray(
            Image.open(row["mask_path"]).convert("L"),
            dtype=np.uint8,
        )

        image, mask = self.transform(image, mask)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask