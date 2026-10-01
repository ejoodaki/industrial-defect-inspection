from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from unet import UNet


ROOT = Path(__file__).resolve().parent

VAL_CSV = ROOT / "splits" / "validation.csv"

CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "hard_negative_patch_poc_best.pt"
)

PATCH_SIZE = 192
TOP_K = 5
THRESHOLD = 0.5

# Targeted but meaningful evaluation:
# all 49 defective + 49 healthy validation samples.
MAX_DEFECTIVE = 49
MAX_HEALTHY = 49

DEVICE = torch.device("cpu")


def load_gray(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    return image


def generate_windows(height, width):
    """
    Generate candidate 192x192 windows.

    No GT information is used here.
    """

    if height < PATCH_SIZE or width < PATCH_SIZE:
        return []

    xs = [0]

    if width > PATCH_SIZE:
        xs.append(width - PATCH_SIZE)

    step = PATCH_SIZE // 4

    ys = list(
        range(
            0,
            height - PATCH_SIZE + 1,
            step,
        )
    )

    last_y = height - PATCH_SIZE

    if last_y not in ys:
        ys.append(last_y)

    windows = []

    for y in ys:
        for x in xs:
            windows.append(
                (x, y)
            )

    return windows


def score_patch(image, x, y):
    """
    Classical candidate score.

    Uses only the image itself.
    """

    patch = image[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE,
    ]

    blurred = cv2.GaussianBlur(
        patch,
        (0, 0),
        sigmaX=5,
    )

    residual = cv2.absdiff(
        patch,
        blurred,
    )

    gx = cv2.Sobel(
        patch,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        patch,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(
        gx,
        gy,
    )

    score = (
        float(residual.mean())
        + 0.25 * float(gradient.mean())
    )

    return score


def get_top_candidates(image):
    candidates = []

    for x, y in generate_windows(
        image.shape[0],
        image.shape[1],
    ):

        score = score_patch(
            image,
            x,
            y,
        )

        candidates.append(
            (
                score,
                x,
                y,
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[:TOP_K]


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

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

    print(
        f"Loaded checkpoint: "
        f"{CHECKPOINT.name}"
    )

    print(
        f"Checkpoint iteration: "
        f"{checkpoint.get('iteration', 'unknown')}"
    )

    print(
        f"Checkpoint positive Dice: "
        f"{checkpoint.get('val_positive_dice', 'unknown')}"
    )

    print()

    return model


def predict_patch(model, patch):
    """
    Run U-Net on one 192x192 grayscale patch.
    """

    patch = (
        patch.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        patch
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    return probability


def infer_image(model, image):
    """
    Full inference pipeline:

        image
          ↓
        top-5 candidates
          ↓
        U-Net
          ↓
        merge into original coordinates
    """

    height, width = image.shape

    candidates = get_top_candidates(
        image
    )

    merged_probability = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    candidate_records = []

    for rank, (score, x, y) in enumerate(
        candidates,
        start=1,
    ):

        patch = image[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ]

        probability = predict_patch(
            model,
            patch,
        )

        # Merge using maximum probability.
        current_region = merged_probability[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ]

        merged_probability[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ] = np.maximum(
            current_region,
            probability,
        )

        candidate_records.append(
            {
                "rank": rank,
                "x": x,
                "y": y,
                "score": score,
                "max_probability": float(
                    probability.max()
                ),
            }
        )

    prediction = (
        merged_probability >= THRESHOLD
    ).astype(np.uint8)

    return (
        prediction,
        merged_probability,
        candidate_records,
    )


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
        2.0 * intersection
        / denominator
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

    return (
        intersection / union
    )


def main():

    print("END-TO-END TWO-STAGE PROTOTYPE")
    print("===============================")
    print(f"Device:       {DEVICE}")
    print(f"Patch size:   {PATCH_SIZE}")
    print(f"Top-K:        {TOP_K}")
    print(f"Threshold:    {THRESHOLD}")
    print()

    df = pd.read_csv(
        VAL_CSV
    )

    defective = df[
        df["has_defect"] == 1
    ].head(MAX_DEFECTIVE)

    healthy = df[
        df["has_defect"] == 0
    ].head(MAX_HEALTHY)

    print(
        f"Defective samples: "
        f"{len(defective)}"
    )

    print(
        f"Healthy samples:   "
        f"{len(healthy)}"
    )

    print()

    model = load_model()

    defective_dices = []
    defective_ious = []

    defective_detected = 0

    healthy_false_positives = 0

    healthy_predicted_pixels = []

    print(
        "Running defective validation..."
    )

    for index, (_, row) in enumerate(
        defective.iterrows(),
        start=1,
    ):

        image = load_gray(
            ROOT / row["image_path"]
        )

        mask = load_gray(
            ROOT / row["mask_path"]
        )

        prediction, probability, candidates = (
            infer_image(
                model,
                image,
            )
        )

        gt = (
            mask > 0
        ).astype(np.uint8)

        dice = dice_score(
            prediction,
            gt,
        )

        iou = iou_score(
            prediction,
            gt,
        )

        defective_dices.append(
            dice
        )

        defective_ious.append(
            iou
        )

        if prediction.sum() > 0:
            defective_detected += 1

        if index <= 5:
            print(
                f"  [{index:02d}] "
                f"{Path(row['image_path']).stem} | "
                f"Dice={dice:.4f} | "
                f"IoU={iou:.4f} | "
                f"PredPixels={prediction.sum()}"
            )

    print()

    print(
        "Running healthy validation..."
    )

    for index, (_, row) in enumerate(
        healthy.iterrows(),
        start=1,
    ):

        image = load_gray(
            ROOT / row["image_path"]
        )

        prediction, probability, candidates = (
            infer_image(
                model,
                image,
            )
        )

        predicted_pixels = int(
            prediction.sum()
        )

        healthy_predicted_pixels.append(
            predicted_pixels
        )

        if predicted_pixels > 0:
            healthy_false_positives += 1

        if index <= 5:
            print(
                f"  [{index:02d}] "
                f"{Path(row['image_path']).stem} | "
                f"PredPixels={predicted_pixels} | "
                f"MaxProb={probability.max():.6f}"
            )

    defective_detection_rate = (
        defective_detected
        / len(defective)
    )

    healthy_false_positive_rate = (
        healthy_false_positives
        / len(healthy)
    )

    print()
    print(
        "END-TO-END RESULTS"
    )
    print(
        "------------------"
    )

    print(
        f"Defective mean Dice:       "
        f"{np.mean(defective_dices):.6f}"
    )

    print(
        f"Defective mean IoU:        "
        f"{np.mean(defective_ious):.6f}"
    )

    print(
        f"Defect detection rate:     "
        f"{defective_detection_rate:.4f}"
    )

    print(
        f"Healthy false-positive rate:"
        f" {healthy_false_positive_rate:.4f}"
    )

    print(
        f"Healthy mean predicted px:  "
        f"{np.mean(healthy_predicted_pixels):.2f}"
    )

    print(
        f"Healthy max predicted px:   "
        f"{np.max(healthy_predicted_pixels)}"
    )

    print()
    print(
        "END-TO-END PROTOTYPE COMPLETED"
    )


if __name__ == "__main__":
    main()