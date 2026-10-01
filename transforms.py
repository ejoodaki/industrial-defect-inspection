import numpy as np
from PIL import Image


class PairedResizePad:
    """
    Resize image and mask while preserving aspect ratio,
    then pad both to a fixed square size.

    Image:
        - bilinear interpolation

    Mask:
        - nearest-neighbor interpolation
        - remains binary (0/1)
    """

    def __init__(self, target_size=384):
        self.target_size = target_size

    def __call__(self, image, mask):
        if image.ndim != 2:
            raise ValueError(
                f"Expected grayscale image with shape (H, W), got {image.shape}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask with shape (H, W), got {mask.shape}"
            )

        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: "
                f"{image.shape} vs {mask.shape}"
            )

        original_height, original_width = image.shape

        scale = min(
            self.target_size / original_height,
            self.target_size / original_width,
        )

        new_height = max(1, round(original_height * scale))
        new_width = max(1, round(original_width * scale))

        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        image_resized = image_pil.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        mask_resized = mask_pil.resize(
            (new_width, new_height),
            Image.Resampling.NEAREST,
        )

        image_resized = np.asarray(image_resized, dtype=np.float32) / 255.0
        mask_resized = np.asarray(mask_resized, dtype=np.float32) / 255.0

        padded_image = np.zeros(
            (self.target_size, self.target_size),
            dtype=np.float32,
        )

        padded_mask = np.zeros(
            (self.target_size, self.target_size),
            dtype=np.float32,
        )

        pad_top = (self.target_size - new_height) // 2
        pad_left = (self.target_size - new_width) // 2

        padded_image[
            pad_top:pad_top + new_height,
            pad_left:pad_left + new_width,
        ] = image_resized

        padded_mask[
            pad_top:pad_top + new_height,
            pad_left:pad_left + new_width,
        ] = mask_resized

        padded_mask = (padded_mask > 0.5).astype(np.float32)

        return padded_image, padded_mask