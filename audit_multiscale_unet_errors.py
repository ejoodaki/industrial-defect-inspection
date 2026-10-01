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
    "outputs/multiscale_unet_error_audit"
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


def evaluate_image(
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

    candidate_info = []

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

        candidate_info.append(
            {
                **candidate,
                "max_probability": float(
                    probability.max()
                ),
                "mean_probability": float(
                    probability.mean()
                ),
                "predicted_pixels": int(
                    (probability >= THRESHOLD).sum()
                ),
            }
        )

    prediction = (
        full_probability >= THRESHOLD
    ).astype(np.uint8)

    return (
        prediction,
        full_probability,
        candidate_info,
    )


def make_overlay(
    image,
    prediction,
    mask,
):
    base = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    # Ground truth = red
    gt = mask.astype(bool)

    # Prediction = green
    pred = prediction.astype(bool)

    overlay = base.copy()

    overlay[gt] = (
        0.5 * overlay[gt]
        + 0.5 * np.array(
            [0, 0, 255],
            dtype=np.float64,
        )
    ).astype(np.uint8)

    overlay[pred] = (
        0.5 * overlay[pred]
        + 0.5 * np.array(
            [0, 255, 0],
            dtype=np.float64,
        )
    ).astype(np.uint8)

    return overlay


def make_candidate_overlay(
    image,
    candidates,
):
    canvas = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        x1 = candidate["x1"]
        y1 = candidate["y1"]
        x2 = candidate["x2"]
        y2 = candidate["y2"]

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            1,
        )

        cv2.putText(
            canvas,
            str(rank),
            (x1 + 3, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


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

    summary = []

    print("=" * 60)
    print("MULTI-SCALE U-NET ERROR AUDIT")
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

        (
            prediction,
            probability,
            candidates,
        ) = evaluate_image(
            model,
            image,
            mask,
        )

        has_defect = bool(mask.any())
        predicted_defect = bool(
            prediction.any()
        )

        if has_defect:
            intersection = np.logical_and(
                prediction > 0,
                mask > 0,
            ).sum()

            union = np.logical_or(
                prediction > 0,
                mask > 0,
            ).sum()

            dice_denominator = (
                prediction.sum()
                + mask.sum()
            )

            dice = (
                1.0
                if dice_denominator == 0
                else (
                    2.0
                    * intersection
                    / dice_denominator
                )
            )

            iou = (
                1.0
                if union == 0
                else intersection / union
            )

            detected = bool(
                intersection > 0
            )

            category = (
                "TP"
                if detected
                else "FN"
            )

        else:
            dice = 1.0
            iou = 1.0
            detected = False

            category = (
                "FP"
                if predicted_defect
                else "TN"
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

        summary.append(
            {
                "image_id": row["image_id"],
                "category": category,
                "dice": dice,
                "iou": iou,
                "predicted_pixels":
                    predicted_pixels,
                "max_probability":
                    max_probability,
                "mean_probability":
                    mean_probability,
            }
        )

        if category in {"FP", "FN"}:
            overlay = make_overlay(
                image,
                prediction,
                mask,
            )

            candidate_overlay = (
                make_candidate_overlay(
                    image,
                    candidates,
                )
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / f"{category}_{row['image_id']}_overlay.png"
                ),
                overlay,
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / f"{category}_{row['image_id']}_candidates.png"
                ),
                candidate_overlay,
            )

        if index % 10 == 0:
            print(
                f"Processed {index}/"
                f"{len(evaluation_rows)}"
            )

    summary_csv = (
        OUTPUT_DIR
        / "error_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "category",
                "dice",
                "iou",
                "predicted_pixels",
                "max_probability",
                "mean_probability",
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 60)
    print("ERROR SUMMARY")
    print("=" * 60)

    categories = ["TP", "FN", "TN", "FP"]

    for category in categories:
        items = [
            row for row in summary
            if row["category"] == category
        ]

        print(
            f"{category}: {len(items)}"
        )

        if items:
            print(
                "  mean predicted px: "
                f"{np.mean([x['predicted_pixels'] for x in items]):.2f}"
            )

            print(
                "  mean max probability: "
                f"{np.mean([x['max_probability'] for x in items]):.6f}"
            )

    print()
    print(
        f"Saved summary: {summary_csv}"
    )
    print(
        f"Saved error images: {OUTPUT_DIR}"
    )
    print()
    print(
        "MULTI-SCALE U-NET ERROR AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()