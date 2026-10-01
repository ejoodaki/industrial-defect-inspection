import numpy as np
import pandas as pd
from PIL import Image

from transforms import PairedResizePad


def test_real_defective_sample_transform():
    csv_path = "splits/train.csv"

    data = pd.read_csv(csv_path)

    defective = data[data["has_defect"] == 1]

    assert len(defective) > 0

    row = defective.iloc[0]

    image = np.asarray(
        Image.open(row["image_path"]).convert("L"),
        dtype=np.uint8,
    )

    mask = np.asarray(
        Image.open(row["mask_path"]).convert("L"),
        dtype=np.uint8,
    )

    original_defect_pixels = int((mask > 0).sum())

    assert original_defect_pixels > 0

    transform = PairedResizePad(target_size=384)

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    # Shape checks
    assert transformed_image.shape == (384, 384)
    assert transformed_mask.shape == (384, 384)

    # Dtype checks
    assert transformed_image.dtype == np.float32
    assert transformed_mask.dtype == np.float32

    # Image range
    assert 0.0 <= transformed_image.min() <= 1.0
    assert 0.0 <= transformed_image.max() <= 1.0

    # Mask must remain binary
    unique_mask_values = set(np.unique(transformed_mask))
    assert unique_mask_values.issubset({0.0, 1.0})

    # Defect must survive transformation
    transformed_defect_pixels = int(
        (transformed_mask > 0.5).sum()
    )

    assert transformed_defect_pixels > 0

    print()
    print("REAL TRANSFORM AUDIT")
    print("--------------------")
    print(f"Sample ID: {row['image_path']}")
    print(f"Original shape: {image.shape}")
    print(f"Transformed shape: {transformed_image.shape}")
    print(f"Original defect pixels: {original_defect_pixels}")
    print(
        f"Transformed defect pixels: "
        f"{transformed_defect_pixels}"
    )
    print(f"Mask values: {sorted(unique_mask_values)}")