import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from transforms import PairedResizePad
from unet import UNet


VAL_CSV = Path("splits/validation.csv")
CHECKPOINT = Path(
    "outputs/checkpoints/hard_negative_patch_poc_best.pt"
)

OUTPUT_DIR = Path(
    "outputs/segmentation_feature_audit"
)

PATCH_SIZES = [128, 192, 256]
STRIDE = 32
TOP_K = 5

MODEL_INPUT_SIZE = 192
THRESHOLD = 0.5

MAX_DEFECTIVE = 49
MAX_HEALTHY = 49

DEVICE = torch.device("cpu")


def load_model():
    model = UNet(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_pair(image_path, mask_path):
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        )

    if image.shape != mask.shape:
        raise RuntimeError(
            f"Shape mismatch: "
            f"{image.shape} vs {mask.shape}"
        )

    mask = (mask > 127).astype(np.uint8)

    return image, mask


def generate_candidates(
    image,
    patch_size,
    stride,
):
    h, w = image.shape

    if h < patch_size or w < patch_size:
        return []

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=3,
    )

    residual = cv2.absdiff(
        image,
        blurred,
    )

    gradient_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(
        gradient_x,
        gradient_y,
    )

    residual = cv2.GaussianBlur(
        residual,
        (0, 0),
        sigmaX=5,
    )

    residual = residual.astype(
        np.float32
    )

    gradient = gradient.astype(
        np.float32
    )

    if residual.max() > 0:
        residual /= residual.max()

    if gradient.max() > 0:
        gradient /= gradient.max()

    score_map = (
        0.75 * residual
        + 0.25 * gradient
    )

    candidates = []

    y_positions = list(
        range(
            0,
            h - patch_size + 1,
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            w - patch_size + 1,
            stride,
        )
    )

    if y_positions[-1] != h - patch_size:
        y_positions.append(
            h - patch_size
        )

    if x_positions[-1] != w - patch_size:
        x_positions.append(
            w - patch_size
        )

    for y1 in y_positions:
        for x1 in x_positions:
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            score = float(
                score_map[
                    y1:y2,
                    x1:x2
                ].mean()
            )

            candidates.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "patch_size": patch_size,
                    "score": score,
                }
            )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates


def predict_candidate(
    model,
    image_patch,
):
    transform = PairedResizePad(
        target_size=MODEL_INPUT_SIZE
    )

    dummy_mask = np.zeros_like(
        image_patch
    )

    transformed_image, _ = transform(
        image_patch,
        dummy_mask,
    )

    tensor = torch.from_numpy(
        transformed_image
    ).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(tensor)

        probability = torch.sigmoid(
            logits
        )[0, 0]

    return probability.cpu().numpy()


def restore_prediction(
    probability,
    original_shape,
):
    height, width = original_shape

    return cv2.resize(
        probability,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def connected_component_features(
    prediction,
):
    binary = (
        prediction > 0
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_bbox_area": 0,
            "largest_component_width": 0,
            "largest_component_height": 0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_index = (
        1 + int(np.argmax(areas))
    )

    largest_area = int(
        stats[
            largest_index,
            cv2.CC_STAT_AREA
        ]
    )

    width = int(
        stats[
            largest_index,
            cv2.CC_STAT_WIDTH
        ]
    )

    height = int(
        stats[
            largest_index,
            cv2.CC_STAT_HEIGHT
        ]
    )

    return {
        "component_count": int(
            num_labels - 1
        ),
        "largest_component_area":
            largest_area,
        "largest_component_bbox_area":
            width * height,
        "largest_component_width":
            width,
        "largest_component_height":
            height,
    }


def bbox_features(
    prediction,
):
    ys, xs = np.where(
        prediction > 0
    )

    if len(xs) == 0:
        return {
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_area": 0,
            "bbox_aspect_ratio": 0.0,
        }

    width = int(
        xs.max() - xs.min() + 1
    )

    height = int(
        ys.max() - ys.min() + 1
    )

    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "bbox_aspect_ratio":
            width / height
            if height > 0
            else 0.0,
    }


def evaluate_image(
    model,
    image,
):
    all_candidates = []

    for patch_size in PATCH_SIZES:
        candidates = generate_candidates(
            image,
            patch_size,
            STRIDE,
        )

        all_candidates.extend(
            candidates[:TOP_K]
        )

    all_candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = all_candidates[:TOP_K]

    full_probability = np.zeros(
        image.shape,
        dtype=np.float32,
    )

    for candidate in selected:
        x1 = candidate["x1"]
        y1 = candidate["y1"]
        x2 = candidate["x2"]
        y2 = candidate["y2"]

        patch = image[
            y1:y2,
            x1:x2,
        ]

        probability = predict_candidate(
            model,
            patch,
        )

        probability = restore_prediction(
            probability,
            patch.shape,
        )

        full_probability[
            y1:y2,
            x1:x2
        ] = np.maximum(
            full_probability[
                y1:y2,
                x1:x2
            ],
            probability,
        )

    prediction = (
        full_probability >= THRESHOLD
    ).astype(np.uint8)

    return prediction, full_probability


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model()

    with open(
        VAL_CSV,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    defective = [
        row for row in rows
        if row["has_defect"] == "1"
    ][:MAX_DEFECTIVE]

    healthy = [
        row for row in rows
        if row["has_defect"] == "0"
    ][:MAX_HEALTHY]

    evaluation_rows = (
        defective + healthy
    )

    results = []

    print("=" * 60)
    print("SEGMENTATION FEATURE AUDIT")
    print("=" * 60)
    print(
        f"Samples: {len(evaluation_rows)}"
    )
    print()

    for index, row in enumerate(
        evaluation_rows,
        start=1,
    ):
        image, mask = load_pair(
            row["image_path"],
            row["mask_path"],
        )

        prediction, probability = (
            evaluate_image(
                model,
                image,
            )
        )

        predicted_pixels = int(
            prediction.sum()
        )

        max_probability = float(
            probability.max()
        )

        mean_probability = float(
            probability.mean()
        )

        image_area = (
            image.shape[0]
            * image.shape[1]
        )

        prediction_density = (
            predicted_pixels
            / image_area
        )

        components = (
            connected_component_features(
                prediction
            )
        )

        bbox = bbox_features(
            prediction
        )

        has_defect = bool(
            mask.any()
        )

        if has_defect:
            intersection = np.logical_and(
                prediction > 0,
                mask > 0,
            ).sum()

            if intersection > 0:
                category = "TP"
            else:
                category = "FN"
        else:
            if predicted_pixels > 0:
                category = "FP"
            else:
                category = "TN"

        results.append(
            {
                "image_id": row["image_id"],
                "category": category,
                "predicted_pixels":
                    predicted_pixels,
                "prediction_density":
                    prediction_density,
                "max_probability":
                    max_probability,
                "mean_probability":
                    mean_probability,
                **components,
                **bbox,
            }
        )

        if index % 10 == 0:
            print(
                f"Processed {index}/"
                f"{len(evaluation_rows)}"
            )

    output_csv = (
        OUTPUT_DIR
        / "segmentation_features.csv"
    )

    fieldnames = [
        "image_id",
        "category",
        "predicted_pixels",
        "prediction_density",
        "max_probability",
        "mean_probability",
        "component_count",
        "largest_component_area",
        "largest_component_bbox_area",
        "largest_component_width",
        "largest_component_height",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "bbox_aspect_ratio",
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    for category in [
        "TP",
        "FN",
        "TN",
        "FP",
    ]:
        items = [
            row for row in results
            if row["category"] == category
        ]

        print()
        print(
            f"{category}: {len(items)}"
        )

        if not items:
            continue

        for feature in [
            "predicted_pixels",
            "prediction_density",
            "max_probability",
            "mean_probability",
            "component_count",
            "largest_component_area",
            "largest_component_bbox_area",
            "largest_component_width",
            "largest_component_height",
            "bbox_width",
            "bbox_height",
            "bbox_area",
            "bbox_aspect_ratio",
        ]:
            values = np.array(
                [
                    row[feature]
                    for row in items
                ],
                dtype=np.float64,
            )

            print(
                f"  {feature}: "
                f"mean={values.mean():.4f} "
                f"median={np.median(values):.4f} "
                f"min={values.min():.4f} "
                f"max={values.max():.4f}"
            )

    print()
    print(
        f"Saved: {output_csv}"
    )
    print()
    print(
        "SEGMENTATION FEATURE AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()