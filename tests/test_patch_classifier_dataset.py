import torch

from patch_classifier_dataset import PatchClassifierDataset


TRAIN_CSVS = [
    "outputs/patch_dataset/positive_train.csv",
    "outputs/patch_dataset/hard_negative_train.csv",
    "outputs/patch_dataset/healthy_negative_train.csv",
]

VALIDATION_CSVS = [
    "outputs/patch_dataset/positive_validation.csv",
    "outputs/patch_dataset/hard_negative_validation.csv",
    "outputs/patch_dataset/healthy_negative_validation.csv",
]


def inspect_dataset(name, csv_paths):
    dataset = PatchClassifierDataset(csv_paths)

    print(f"{name} DATASET")
    print("-" * 30)
    print(f"Samples: {len(dataset)}")

    for index in [
        0,
        len(dataset) // 2,
        len(dataset) - 1,
    ]:
        patch, label = dataset[index]

        print(
            f"Index {index}: "
            f"patch={tuple(patch.shape)}, "
            f"dtype={patch.dtype}, "
            f"label={label.item():.0f}, "
            f"range="
            f"{patch.min().item():.4f}"
            f"-"
            f"{patch.max().item():.4f}"
        )

        assert patch.shape == (1, 192, 192)
        assert patch.dtype == torch.float32
        assert 0.0 <= patch.min().item() <= 1.0
        assert 0.0 <= patch.max().item() <= 1.0
        assert label.item() in {0.0, 1.0}

    print()


def main():
    inspect_dataset(
        "TRAIN",
        TRAIN_CSVS,
    )

    inspect_dataset(
        "VALIDATION",
        VALIDATION_CSVS,
    )

    print(
        "PATCH CLASSIFIER DATASET PASSED"
    )


if __name__ == "__main__":
    main()