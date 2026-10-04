import numpy as np

from transforms import PairedResizePad


def test_paired_resize_pad_shapes_and_ranges():
    image = np.full((645, 229), 128, dtype=np.uint8)
    mask = np.zeros((645, 229), dtype=np.uint8)

    mask[300:340, 100:130] = 255

    transform = PairedResizePad(target_size=384)

    transformed_image, transformed_mask = transform(image, mask)

    assert transformed_image.shape == (384, 384)
    assert transformed_mask.shape == (384, 384)

    assert transformed_image.dtype == np.float32
    assert transformed_mask.dtype == np.float32

    assert 0.0 <= transformed_image.min() <= 1.0
    assert 0.0 <= transformed_image.max() <= 1.0

    assert set(np.unique(transformed_mask)).issubset({0.0, 1.0})

    assert transformed_mask.sum() > 0


def test_image_and_mask_keep_same_geometry():
    image = np.zeros((640, 200), dtype=np.uint8)
    mask = np.zeros((640, 200), dtype=np.uint8)

    image[250:350, 70:100] = 255
    mask[250:350, 70:100] = 255

    transform = PairedResizePad(target_size=384)

    transformed_image, transformed_mask = transform(image, mask)

    image_foreground = transformed_image > 0.5
    mask_foreground = transformed_mask > 0.5

    assert np.array_equal(
        image_foreground,
        mask_foreground,
    )