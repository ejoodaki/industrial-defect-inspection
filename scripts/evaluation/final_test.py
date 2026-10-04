from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from transforms import PairedResizePad
from unet import UNet


TEST_DIR = Path("test")
MODEL_PATH = Path(
    "outputs/checkpoints/hard_negative_patch_poc_best.pt"
)
RF_PATH = Path(
    "outputs/decision_layer/random_forest_classifier.joblib"
)

OUTPUT_DIR = Path("outputs/final_test")
PREDICTIONS_PATH = OUTPUT_DIR / "final_test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "final_test_metrics.json"

PATCH_SIZES = [128, 192, 256]
STRIDE = 32

MODEL_INPUT_SIZE = 192
U_NET_THRESHOLD = 0.5
RF_THRESHOLD = 0.30

EXPECTED_SAMPLES = 1004
EXPECTED_FIRST_ID = 20000
EXPECTED_LAST_ID = 21003

DEVICE = torch.device("cpu")

FEATURE_COLUMNS = [
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


def load_model():
    model = UNet(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
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
            f"Shape mismatch: {image_path.name} "
            f"{image.shape} vs {mask_path.name} {mask.shape}"
        )

    mask = (mask > 127).astype(np.uint8)

    return image, mask


def generate_candidates(
    image,
    patch_size,
    stride,
):
    height, width = image.shape

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=3,
    )

    residual = cv2.absdiff(
        image,
        blurred,
    )

    residual = cv2.GaussianBlur(
        residual,
        (0, 0),
        sigmaX=5,
    )

    grad_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    grad_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(
        grad_x,
        grad_y,
    )

    residual = residual.astype(
        np.float32
    )

    residual_max = residual.max()

    if residual_max > 0:
        residual /= residual_max

    gradient_max = gradient.max()

    if gradient_max > 0:
        gradient /= gradient_max

    score_map = (
        0.75 * residual
        + 0.25 * gradient
    )

    candidates = []

    y_positions = list(
        range(
            0,
            max(height - patch_size + 1, 1),
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            max(width - patch_size + 1, 1),
            stride,
        )
    )

    last_y = height - patch_size
    last_x = width - patch_size

    if last_y >= 0 and last_y not in y_positions:
        y_positions.append(last_y)

    if last_x >= 0 and last_x not in x_positions:
        x_positions.append(last_x)

    for y in y_positions:
        for x in x_positions:
            y2 = min(
                y + patch_size,
                height,
            )

            x2 = min(
                x + patch_size,
                width,
            )

            patch_score = score_map[
                y:y2,
                x:x2,
            ].mean()

            candidates.append(
                (
                    float(patch_score),
                    x,
                    y,
                    x2,
                    y2,
                )
            )

    candidates.sort(
        key=lambda item: item[0],
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
        image_patch,
        dtype=np.uint8,
    )

    transformed_image, _ = transform(
        image_patch,
        dummy_mask,
    )

    tensor = torch.from_numpy(
        transformed_image
    ).float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0).to(DEVICE)

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
    binary = prediction.astype(
        np.uint8
    )

    count, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if count <= 1:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_bbox_area": 0,
            "largest_component_width": 0,
            "largest_component_height": 0,
        }

    component_stats = stats[1:]

    largest = component_stats[
        np.argmax(component_stats[:, cv2.CC_STAT_AREA])
    ]

    x = largest[cv2.CC_STAT_LEFT]
    y = largest[cv2.CC_STAT_TOP]
    width = largest[cv2.CC_STAT_WIDTH]
    height = largest[cv2.CC_STAT_HEIGHT]
    area = largest[cv2.CC_STAT_AREA]

    return {
        "component_count": int(count - 1),
        "largest_component_area": int(area),
        "largest_component_bbox_area": int(
            width * height
        ),
        "largest_component_width": int(width),
        "largest_component_height": int(height),
    }


def bbox_features(prediction):
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

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    width = int(
        x_max - x_min + 1
    )

    height = int(
        y_max - y_min + 1
    )

    area = width * height

    aspect_ratio = (
        width / height
        if height > 0
        else 0.0
    )

    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": int(area),
        "bbox_aspect_ratio": float(
            aspect_ratio
        ),
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
            candidates
        )

    all_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    full_probability = np.zeros(
        image.shape,
        dtype=np.float32,
    )

    for _, x, y, x2, y2 in all_candidates:
        image_patch = image[
            y:y2,
            x:x2,
        ]

        probability = predict_candidate(
            model,
            image_patch,
        )

        restored = restore_prediction(
            probability,
            image_patch.shape,
        )

        full_probability[
            y:y2,
            x:x2,
        ] = np.maximum(
            full_probability[
                y:y2,
                x:x2,
            ],
            restored,
        )

    prediction = (
        full_probability >= U_NET_THRESHOLD
    ).astype(np.uint8)

    return (
        prediction,
        full_probability,
    )


def collect_test_pairs():
    image_paths = sorted(
        [
            path
            for path in TEST_DIR.glob("*.png")
            if not path.name.endswith("_GT.png")
        ],
        key=lambda path: int(path.stem),
    )

    if len(image_paths) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} test images, "
            f"found {len(image_paths)}"
        )

    image_ids = [
        int(path.stem)
        for path in image_paths
    ]

    expected_ids = list(
        range(
            EXPECTED_FIRST_ID,
            EXPECTED_LAST_ID + 1,
        )
    )

    if image_ids != expected_ids:
        raise RuntimeError(
            "Test image IDs are not the expected "
            "continuous range 20000-21003."
        )

    pairs = []

    for image_path in image_paths:
        image_id = image_path.stem

        mask_path = (
            TEST_DIR
            / f"{image_id}_GT.png"
        )

        if not mask_path.exists():
            raise RuntimeError(
                f"Missing GT mask: {mask_path}"
            )

        pairs.append(
            (
                image_id,
                image_path,
                mask_path,
            )
        )

    if len(pairs) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected number of test pairs."
        )

    return pairs


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "FINAL TEST - FROZEN PIPELINE"
    )
    print(
        "============================"
    )
    print(
        f"Test directory: {TEST_DIR}"
    )
    print(
        f"Expected samples: {EXPECTED_SAMPLES}"
    )
    print(
        f"U-Net threshold: {U_NET_THRESHOLD:.2f}"
    )
    print(
        f"RF threshold: {RF_THRESHOLD:.2f}"
    )
    print()

    pairs = collect_test_pairs()

    print(
        f"Verified test pairs: {len(pairs)}"
    )

    print(
        "\nLoading U-Net..."
    )

    model = load_model()

    print(
        "Loading Random Forest..."
    )

    rf_model = joblib.load(
        RF_PATH
    )

    print(
        "Running frozen test pipeline..."
    )

    rows = []

    for index, (
        image_id,
        image_path,
        mask_path,
    ) in enumerate(
        pairs,
        start=1,
    ):
        image, mask = load_pair(
            image_path,
            mask_path,
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

        feature_row = {
            "predicted_pixels": predicted_pixels,
            "prediction_density": float(
                prediction_density
            ),
            "max_probability": float(
                probability.max()
            ),
            "mean_probability": float(
                probability.mean()
            ),
            **components,
            **bbox,
        }

        feature_vector = np.array(
            [
                feature_row[name]
                for name in FEATURE_COLUMNS
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        rf_probability = float(
            rf_model.predict_proba(
                feature_vector
            )[0, 1]
        )

        rf_prediction = int(
            rf_probability >= RF_THRESHOLD
        )

        has_defect = int(
            mask.any()
        )

        if has_defect and rf_prediction:
            category = "TP"
        elif has_defect and not rf_prediction:
            category = "FN"
        elif not has_defect and rf_prediction:
            category = "FP"
        else:
            category = "TN"

        row = {
            "image_id": image_id,
            "has_defect": has_defect,
            "prediction": rf_prediction,
            "category": category,
            "rf_probability": rf_probability,
            **feature_row,
        }

        rows.append(row)

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(pairs)}"
            )

    y_true = np.array(
        [row["has_defect"] for row in rows]
    )

    y_pred = np.array(
        [row["prediction"] for row in rows]
    )

    y_score = np.array(
        [row["rf_probability"] for row in rows]
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "dataset": "KolektorSDD2_test",
        "samples": int(len(rows)),
        "defective_samples": int(
            y_true.sum()
        ),
        "healthy_samples": int(
            len(y_true) - y_true.sum()
        ),
        "u_net_threshold": U_NET_THRESHOLD,
        "rf_threshold": RF_THRESHOLD,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "fpr": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
    }

    import csv

    with open(
        PREDICTIONS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print()
    print(
        "FINAL TEST RESULTS"
    )
    print(
        "=================="
    )

    for key in [
        "samples",
        "defective_samples",
        "healthy_samples",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "roc_auc",
        "average_precision",
    ]:
        print(
            f"{key}: {metrics[key]}"
        )

    print()
    print(
        f"Saved: {PREDICTIONS_PATH}"
    )
    print(
        f"Saved: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()