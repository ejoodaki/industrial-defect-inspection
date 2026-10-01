import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from transforms import PairedResizePad
from unet import UNet


# ============================================================
# Configuration
# ============================================================

VAL_CSV = Path("splits/validation.csv")
CHECKPOINT = Path(
    "outputs/checkpoints/hard_negative_patch_poc_best.pt"
)
OUTPUT_DIR = Path("outputs/multiscale_unet")

PATCH_SIZES = [128, 192, 256]
STRIDE = 32
TOP_K = 5
MODEL_INPUT_SIZE = 192
THRESHOLD = 0.5

MAX_DEFECTIVE = 49
MAX_HEALTHY = 49

DEVICE = torch.device("cpu")


# ============================================================
# Model
# ============================================================

def load_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ============================================================
# Image / mask helpers
# ============================================================

def load_pair(image_path, mask_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    if image.shape != mask.shape:
        raise RuntimeError(
            f"Shape mismatch: {image.shape} vs {mask.shape}"
        )

    mask = (mask > 127).astype(np.uint8)

    return image, mask


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


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates(image, patch_size, stride):
    h, w = image.shape

    if h < patch_size or w < patch_size:
        return []

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=3,
    )

    residual = cv2.absdiff(image, blurred)

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

    residual = residual.astype(np.float32)
    gradient = gradient.astype(np.float32)

    residual_max = residual.max()
    gradient_max = gradient.max()

    if residual_max > 0:
        residual /= residual_max

    if gradient_max > 0:
        gradient /= gradient_max

    score_map = (
        0.75 * residual
        + 0.25 * gradient
    )

    candidates = []

    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))

    if y_positions[-1] != h - patch_size:
        y_positions.append(h - patch_size)

    if x_positions[-1] != w - patch_size:
        x_positions.append(w - patch_size)

    for y1 in y_positions:
        for x1 in x_positions:
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            score = float(
                score_map[y1:y2, x1:x2].mean()
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

    return candidates[:TOP_K]


# ============================================================
# U-Net prediction
# ============================================================

def predict_candidate(model, image_patch):
    transform = PairedResizePad(
        target_size=MODEL_INPUT_SIZE
    )

    dummy_mask = np.zeros_like(image_patch)

    transformed_image, _ = transform(
        image_patch,
        dummy_mask,
    )

    tensor = torch.from_numpy(
        transformed_image
    ).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0]

    probability = probability.cpu().numpy()

    return probability


def restore_prediction(
    probability,
    candidate_shape,
):
    height, width = candidate_shape

    restored = cv2.resize(
        probability,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    return restored


# ============================================================
# Full-image merge
# ============================================================

def merge_predictions(
    image_shape,
    candidate_predictions,
):
    h, w = image_shape

    full_probability = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for item in candidate_predictions:
        x1 = item["x1"]
        y1 = item["y1"]
        x2 = item["x2"]
        y2 = item["y2"]

        prediction = item["prediction"]

        region = full_probability[
            y1:y2,
            x1:x2,
        ]

        np.maximum(
            region,
            prediction,
            out=region,
        )

    return full_probability


# ============================================================
# Metrics
# ============================================================

def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    )


def iou_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


def contains_prediction(prediction, target):
    return bool(
        np.logical_and(
            prediction > 0,
            target > 0,
        ).any()
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_sample(
    model,
    image,
    mask,
):
    all_candidates = []

    for patch_size in PATCH_SIZES:
        candidates = generate_candidates(
            image,
            patch_size,
            STRIDE,
        )

        all_candidates.extend(candidates)

    all_candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected_candidates = all_candidates[:TOP_K]

    candidate_predictions = []

    for candidate in selected_candidates:
        x1 = candidate["x1"]
        y1 = candidate["y1"]
        x2 = candidate["x2"]
        y2 = candidate["y2"]

        image_patch = image[
            y1:y2,
            x1:x2,
        ]

        probability = predict_candidate(
            model,
            image_patch,
        )

        probability = restore_prediction(
            probability,
            image_patch.shape,
        )

        candidate_predictions.append(
            {
                **candidate,
                "prediction": probability,
            }
        )

    full_probability = merge_predictions(
        image.shape,
        candidate_predictions,
    )

    prediction_mask = (
        full_probability >= THRESHOLD
    ).astype(np.uint8)

    return (
        prediction_mask,
        full_probability,
        selected_candidates,
    )


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, checkpoint = load_model()

    print("=" * 60)
    print("MULTI-SCALE CANDIDATE -> U-NET EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Checkpoint iteration: {checkpoint['iteration']}")
    print(
        f"Checkpoint positive Dice: "
        f"{checkpoint['val_positive_dice']:.6f}"
    )
    print(
        f"Checkpoint detection rate: "
        f"{checkpoint['detection_rate']:.4f}"
    )
    print(
        f"Checkpoint hard-negative FP: "
        f"{checkpoint['hard_negative_false_positive_rate']:.4f}"
    )
    print(f"Candidate sizes: {PATCH_SIZES}")
    print(f"Candidate stride: {STRIDE}")
    print(f"Top-K: {TOP_K}")
    print(f"U-Net input size: {MODEL_INPUT_SIZE}")
    print(f"Threshold: {THRESHOLD}")
    print()

    rows = []

    with open(
        VAL_CSV,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = list(csv.DictReader(f))

    defective_rows = [
        row for row in reader
        if row["has_defect"] == "1"
    ][:MAX_DEFECTIVE]

    healthy_rows = [
        row for row in reader
        if row["has_defect"] == "0"
    ][:MAX_HEALTHY]

    evaluation_rows = (
        defective_rows
        + healthy_rows
    )

    print(
        f"Defective samples: {len(defective_rows)}"
    )
    print(
        f"Healthy samples: {len(healthy_rows)}"
    )
    print()

    defective_dice = []
    defective_iou = []
    defective_detection = []

    healthy_fp = []
    healthy_pred_pixels = []

    for index, row in enumerate(
        evaluation_rows,
        start=1,
    ):
        image, mask = load_pair(
            row["image_path"],
            row["mask_path"],
        )

        prediction, probability, candidates = (
            evaluate_sample(
                model,
                image,
                mask,
            )
        )

        is_defective = bool(mask.any())

        if is_defective:
            dice = dice_score(
                prediction,
                mask,
            )

            iou = iou_score(
                prediction,
                mask,
            )

            detected = contains_prediction(
                prediction,
                mask,
            )

            defective_dice.append(dice)
            defective_iou.append(iou)
            defective_detection.append(
                float(detected)
            )

            rows.append(
                {
                    "image_id": row["image_id"],
                    "has_defect": 1,
                    "dice": dice,
                    "iou": iou,
                    "detected": int(detected),
                    "predicted_pixels": int(
                        prediction.sum()
                    ),
                }
            )

        else:
            false_positive = int(
                prediction.any()
            )

            healthy_fp.append(
                float(false_positive)
            )

            healthy_pred_pixels.append(
                int(prediction.sum())
            )

            rows.append(
                {
                    "image_id": row["image_id"],
                    "has_defect": 0,
                    "dice": 1.0,
                    "iou": 1.0,
                    "detected": 0,
                    "predicted_pixels": int(
                        prediction.sum()
                    ),
                }
            )

        if index % 10 == 0:
            print(
                f"Processed {index}/"
                f"{len(evaluation_rows)}"
            )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if defective_dice:
        print(
            f"Defective mean Dice: "
            f"{np.mean(defective_dice):.6f}"
        )

        print(
            f"Defective mean IoU: "
            f"{np.mean(defective_iou):.6f}"
        )

        print(
            f"Defect detection rate: "
            f"{np.mean(defective_detection):.4f}"
        )

    if healthy_fp:
        print(
            f"Healthy false-positive rate: "
            f"{np.mean(healthy_fp):.4f}"
        )

        print(
            f"Healthy mean predicted px: "
            f"{np.mean(healthy_pred_pixels):.2f}"
        )

        print(
            f"Healthy max predicted px: "
            f"{np.max(healthy_pred_pixels)}"
        )

    output_csv = (
        OUTPUT_DIR
        / "multiscale_unet_results.csv"
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "has_defect",
                "dice",
                "iou",
                "detected",
                "predicted_pixels",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {output_csv}")
    print()
    print(
        "MULTI-SCALE U-NET EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()