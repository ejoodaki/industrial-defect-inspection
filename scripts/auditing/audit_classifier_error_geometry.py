from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(".")
ERROR_DIR = ROOT / "outputs" / "patch_classifier" / "error_analysis"

FN_FILE = ERROR_DIR / "false_negatives.csv"
FP_FILE = ERROR_DIR / "false_positives.csv"

OUTPUT_FILE = ERROR_DIR / "error_geometry_audit.csv"

PATCH_SIZE = 192


def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return mask


def get_defect_geometry(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return {
            "defect_pixels": 0,
            "defect_ratio": 0.0,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_area": 0,
            "bbox_aspect_ratio": 0.0,
            "bbox_x1": -1,
            "bbox_y1": -1,
            "bbox_x2": -1,
            "bbox_y2": -1,
        }

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    bbox_width = x2 - x1 + 1
    bbox_height = y2 - y1 + 1
    bbox_area = bbox_width * bbox_height

    return {
        "defect_pixels": int(len(xs)),
        "defect_ratio": float(len(xs) / mask.size),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "bbox_aspect_ratio": float(bbox_width / bbox_height),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
    }


def get_patch_coordinates(row):
    image_path = ROOT / row["image_path"]
    mask_path = ROOT / row["mask_path"]

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = load_mask(mask_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape

    if row["patch_type"] == "positive":
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            raise ValueError(
                f"Positive sample has no defect pixels: {row['image_id']}"
            )

        center_x = float(np.median(xs))
        center_y = float(np.median(ys))

        half = PATCH_SIZE // 2

        x1 = int(round(center_x - half))
        y1 = int(round(center_y - half))

        x1 = max(0, min(x1, width - PATCH_SIZE))
        y1 = max(0, min(y1, height - PATCH_SIZE))

    else:
        # Reconstruct negative patch from the same dataset artifact.
        if row["patch_type"] in {"healthy_negative", "hard_negative"}:
            dataset_csv = ROOT / "outputs" / "patch_dataset"

            if row["patch_type"] == "healthy_negative":
                source_csv = dataset_csv / "healthy_negative_validation.csv"
            else:
                source_csv = dataset_csv / "hard_negative_validation.csv"

            source_df = pd.read_csv(source_csv)

            match = source_df[
                source_df["image_id"] == row["image_id"]
            ]

            if len(match) == 0:
                raise ValueError(
                    f"Could not find patch coordinates for "
                    f"{row['image_id']} ({row['patch_type']})"
                )

            # There can be multiple patches from the same image.
            # Use dataset_index to select the exact dataset row.
            local_matches = match.reset_index(drop=True)

            if len(local_matches) == 1:
                source_row = local_matches.iloc[0]
            else:
                # Match against the validation dataset ordering.
                source_row = local_matches.iloc[
                    int(row["dataset_index"]) % len(local_matches)
                ]

            x1 = int(source_row["x1"])
            y1 = int(source_row["y1"])

        else:
            raise ValueError(
                f"Unknown patch type: {row['patch_type']}"
            )

    x2 = x1 + PATCH_SIZE
    y2 = y1 + PATCH_SIZE

    return (
        image,
        mask,
        x1,
        y1,
        x2,
        y2,
    )


def analyze_row(row):
    image, mask, x1, y1, x2, y2 = get_patch_coordinates(row)

    patch_mask = mask[y1:y2, x1:x2]

    if patch_mask.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"Invalid patch shape for {row['image_id']}: "
            f"{patch_mask.shape}"
        )

    geometry = get_defect_geometry(patch_mask)

    patch_height, patch_width = patch_mask.shape

    if geometry["defect_pixels"] > 0:
        bbox_x1 = geometry["bbox_x1"]
        bbox_y1 = geometry["bbox_y1"]
        bbox_x2 = geometry["bbox_x2"]
        bbox_y2 = geometry["bbox_y2"]

        edge_distance = min(
            bbox_x1,
            bbox_y1,
            patch_width - 1 - bbox_x2,
            patch_height - 1 - bbox_y2,
        )
    else:
        edge_distance = -1

    result = {
        "dataset_index": int(row["dataset_index"]),
        "image_id": int(row["image_id"]),
        "patch_type": row["patch_type"],
        "label": int(row["label"]),
        "prediction": int(row["prediction"]),
        "probability": float(row["probability"]),
        "patch_x1": x1,
        "patch_y1": y1,
        "patch_x2": x2,
        "patch_y2": y2,
        "patch_edge_distance": int(edge_distance),
    }

    result.update(geometry)

    return result


def main():
    print("CLASSIFIER ERROR GEOMETRY AUDIT")
    print("--------------------------------")
    print()

    fn_df = pd.read_csv(FN_FILE)
    fp_df = pd.read_csv(FP_FILE)

    errors = pd.concat(
        [fn_df, fp_df],
        ignore_index=True,
    )

    results = []

    for _, row in errors.iterrows():
        results.append(analyze_row(row))

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["patch_type", "probability"],
        ascending=[True, True],
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Total errors: {len(result_df)}")
    print()

    print("ERROR COUNTS")
    print("------------")
    print(
        result_df["patch_type"]
        .value_counts()
        .to_string()
    )
    print()

    print("FALSE NEGATIVES")
    print("----------------")
    fn_result = result_df[result_df["label"] == 1]

    print(
        fn_result[
            [
                "image_id",
                "probability",
                "defect_pixels",
                "defect_ratio",
                "bbox_width",
                "bbox_height",
                "bbox_area",
                "bbox_aspect_ratio",
                "patch_edge_distance",
            ]
        ].to_string(index=False)
    )

    print()

    print("FALSE POSITIVES")
    print("----------------")
    fp_result = result_df[result_df["label"] == 0]

    print(
        fp_result[
            [
                "image_id",
                "patch_type",
                "probability",
                "defect_pixels",
                "defect_ratio",
                "bbox_width",
                "bbox_height",
                "bbox_area",
                "bbox_aspect_ratio",
                "patch_edge_distance",
            ]
        ].to_string(index=False)
    )

    print()

    print("SUMMARY")
    print("-------")

    print(
        f"FN defect pixels median: "
        f"{fn_result['defect_pixels'].median():.2f}"
    )

    print(
        f"FN defect ratio median: "
        f"{fn_result['defect_ratio'].median():.6f}"
    )

    print(
        f"FN bbox area median: "
        f"{fn_result['bbox_area'].median():.2f}"
    )

    print(
        f"FN edge distance median: "
        f"{fn_result['patch_edge_distance'].median():.2f}"
    )

    print()

    print(
        f"FP probability median: "
        f"{fp_result['probability'].median():.6f}"
    )

    print(
        f"Hard-negative FP count: "
        f"{len(fp_result[fp_result['patch_type'] == 'hard_negative'])}"
    )

    print(
        f"Healthy FP count: "
        f"{len(fp_result[fp_result['patch_type'] == 'healthy_negative'])}"
    )

    print()

    print(f"Saved: {OUTPUT_FILE}")
    print()
    print("CLASSIFIER ERROR GEOMETRY AUDIT PASSED")


if __name__ == "__main__":
    main()