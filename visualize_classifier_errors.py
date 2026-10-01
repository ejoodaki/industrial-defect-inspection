
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patch_classifier_dataset import PatchClassifierDataset


VALIDATION_CSVS = [
    "outputs/patch_dataset/positive_validation.csv",
    "outputs/patch_dataset/healthy_negative_validation.csv",
    "outputs/patch_dataset/hard_negative_validation.csv",
]

ERROR_ANALYSIS_DIR = Path(
    "outputs/patch_classifier/error_analysis"
)

OUTPUT_DIR = (
    ERROR_ANALYSIS_DIR / "visualizations"
)


def load_dataset():
    return PatchClassifierDataset(
        VALIDATION_CSVS
    )


def get_positive_patch_and_mask(
    dataset,
    sample,
):
    image = dataset.load_grayscale(
        sample["image_path"]
    )

    mask = dataset.load_grayscale(
        sample["mask_path"]
    )

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError(
            "Positive sample contains no defect pixels."
        )

    center_x = float(np.median(xs))
    center_y = float(np.median(ys))

    patch_size = 192
    half = patch_size // 2

    x1 = int(
        round(center_x - half)
    )

    y1 = int(
        round(center_y - half)
    )

    height, width = image.shape

    x1 = max(
        0,
        min(
            x1,
            width - patch_size,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            height - patch_size,
        ),
    )

    x2 = x1 + patch_size
    y2 = y1 + patch_size

    patch = image[
        y1:y2,
        x1:x2,
    ]

    mask_patch = mask[
        y1:y2,
        x1:x2,
    ]

    return patch, mask_patch


def get_negative_patch(
    dataset,
    sample,
):
    image = dataset.load_grayscale(
        sample["image_path"]
    )

    x1 = int(sample["x1"])
    y1 = int(sample["y1"])
    x2 = int(sample["x2"])
    y2 = int(sample["y2"])

    patch = image[
        y1:y2,
        x1:x2,
    ]

    if patch.shape != (
        192,
        192,
    ):
        raise ValueError(
            f"Invalid patch shape: "
            f"{patch.shape}"
        )

    return patch, None


def save_visualization(
    sample,
    patch,
    mask_patch,
    error_type,
    output_path,
):
    label = int(
        sample["label"]
    )

    prediction = int(
        sample["prediction"]
    )

    probability = float(
        sample["probability"]
    )

    image_id = sample["image_id"]
    patch_type = sample["patch_type"]

    if error_type == "FN":
        title = (
            f"FALSE NEGATIVE | "
            f"ID={image_id} | "
            f"Type={patch_type} | "
            f"GT=DEFECT | "
            f"Pred=HEALTHY | "
            f"P={probability:.4f}"
        )
    else:
        title = (
            f"FALSE POSITIVE | "
            f"ID={image_id} | "
            f"Type={patch_type} | "
            f"GT=HEALTHY | "
            f"Pred=DEFECT | "
            f"P={probability:.4f}"
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4),
    )

    axes[0].imshow(
        patch,
        cmap="gray",
    )

    axes[0].set_title(
        "Classifier Patch"
    )

    axes[0].axis("off")

    if mask_patch is not None:
        axes[1].imshow(
            patch,
            cmap="gray",
        )

        axes[1].imshow(
            mask_patch > 0,
            alpha=0.35,
        )

        axes[1].set_title(
            "Patch + Ground Truth"
        )

    else:
        axes[1].imshow(
            patch,
            cmap="gray",
        )

        axes[1].set_title(
            "Healthy Ground Truth"
        )

    axes[1].axis("off")

    figure.suptitle(
        title,
        fontsize=10,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def process_errors(
    dataframe,
    dataset,
    error_type,
    output_dir,
):
    for _, row in dataframe.iterrows():

        dataset_index = int(
            row["dataset_index"]
        )

        if dataset_index >= len(dataset):
            raise IndexError(
                f"Invalid dataset index: "
                f"{dataset_index}"
            )

        sample = dict(
            dataset.samples[
                dataset_index
            ]
        )

        sample.update(
            {
                "label": row["label"],
                "prediction": row["prediction"],
                "probability": row["probability"],
                "image_id": row["image_id"],
                "patch_type": row["patch_type"],
            }
        )

        if (
            sample["patch_type"]
            == "positive"
        ):
            patch, mask_patch = (
                get_positive_patch_and_mask(
                    dataset,
                    sample,
                )
            )

        else:
            patch, mask_patch = (
                get_negative_patch(
                    dataset,
                    sample,
                )
            )

        filename = (
            f"{error_type}_"
            f"{sample['image_id']}_"
            f"idx_{dataset_index}.png"
        )

        save_visualization(
            sample,
            patch,
            mask_patch,
            error_type,
            output_dir / filename,
        )


def main():
    print(
        "PATCH CLASSIFIER ERROR "
        "VISUALIZATION"
    )
    print(
        "------------------------------------"
    )

    false_negatives_path = (
        ERROR_ANALYSIS_DIR
        / "false_negatives.csv"
    )

    false_positives_path = (
        ERROR_ANALYSIS_DIR
        / "false_positives.csv"
    )

    if not false_negatives_path.exists():
        raise FileNotFoundError(
            f"Missing: "
            f"{false_negatives_path}"
        )

    if not false_positives_path.exists():
        raise FileNotFoundError(
            f"Missing: "
            f"{false_positives_path}"
        )

    false_negatives = pd.read_csv(
        false_negatives_path
    )

    false_positives = pd.read_csv(
        false_positives_path
    )

    print()
    print(
        f"False negatives: "
        f"{len(false_negatives)}"
    )

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    dataset = load_dataset()

    fn_dir = (
        OUTPUT_DIR / "false_negatives"
    )

    fp_dir = (
        OUTPUT_DIR / "false_positives"
    )

    fn_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    process_errors(
        false_negatives,
        dataset,
        "FN",
        fn_dir,
    )

    process_errors(
        false_positives,
        dataset,
        "FP",
        fp_dir,
    )

    print()
    print("Saved:")
    print(fn_dir)
    print(fp_dir)

    print()
    print(
        "Generated FN visualizations: "
        f"{len(false_negatives)}"
    )

    print(
        "Generated FP visualizations: "
        f"{len(false_positives)}"
    )

    print()
    print(
        "PATCH CLASSIFIER ERROR "
        "VISUALIZATION PASSED"
    )


if __name__ == "__main__":
    main()
