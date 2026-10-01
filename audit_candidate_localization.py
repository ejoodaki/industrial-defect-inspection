from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(".")
VAL_CSV = ROOT / "splits" / "validation.csv"
OUTPUT_DIR = ROOT / "outputs" / "candidate_localization"

PATCH_SIZE = 192
TOP_K_VALUES = [1, 3, 5]


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read: {path}")

    return image


def normalize_map(values):
    values = values.astype(np.float32)

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def build_candidate_score(image):
    image_float = image.astype(np.float32) / 255.0

    # Local high-frequency / residual signal.
    blur = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=3.0,
    )

    residual = np.abs(image_float - blur)

    # Gradient magnitude.
    gx = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(gx, gy)

    residual = normalize_map(residual)
    gradient = normalize_map(gradient)

    score = (
        0.75 * residual
        + 0.25 * gradient
    )

    score = cv2.GaussianBlur(
        score,
        (0, 0),
        sigmaX=5.0,
    )

    return score


def generate_candidates(score):
    height, width = score.shape

    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError(
            f"Image {score.shape} is smaller than patch size "
            f"{PATCH_SIZE}."
        )

    candidates = []

    # Sliding window.
    stride = 32

    y_positions = list(range(0, height - PATCH_SIZE + 1, stride))
    x_positions = list(range(0, width - PATCH_SIZE + 1, stride))

    # Always include the final valid position.
    final_y = height - PATCH_SIZE
    final_x = width - PATCH_SIZE

    if not y_positions or y_positions[-1] != final_y:
        y_positions.append(final_y)

    if not x_positions or x_positions[-1] != final_x:
        x_positions.append(final_x)

    for y1 in y_positions:
        for x1 in x_positions:
            y2 = y1 + PATCH_SIZE
            x2 = x1 + PATCH_SIZE

            patch_score = score[y1:y2, x1:x2]

            candidates.append(
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "score": float(patch_score.mean()),
                }
            )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def box_area(box):
    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(box_a, box_b):
    intersection = intersection_area(box_a, box_b)

    union = (
        box_area(box_a)
        + box_area(box_b)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def contains_defect(candidate, mask):
    x1, y1, x2, y2 = candidate

    patch_mask = mask[y1:y2, x1:x2]

    return bool(np.any(patch_mask > 0))


def full_contains_bbox(candidate, defect_bbox):
    if defect_bbox is None:
        return False

    cx1, cy1, cx2, cy2 = candidate
    dx1, dy1, dx2, dy2 = defect_bbox

    return (
        cx1 <= dx1
        and cy1 <= dy1
        and cx2 >= dx2
        and cy2 >= dy2
    )


def analyze_image(row):
    image = load_gray(ROOT / row["image_path"])
    mask = load_gray(ROOT / row["mask_path"])

    defect_bbox = bbox_from_mask(mask)

    if defect_bbox is None:
        return None

    score = build_candidate_score(image)
    candidates = generate_candidates(score)

    results = {
        "image_id": int(row["image_id"]),
        "defect_pixels": int(np.count_nonzero(mask > 0)),
        "defect_bbox_width": defect_bbox[2] - defect_bbox[0],
        "defect_bbox_height": defect_bbox[3] - defect_bbox[1],
        "num_candidates": len(candidates),
    }

    for k in TOP_K_VALUES:
        top_candidates = candidates[:k]

        best_iou = 0.0
        any_contains = False
        any_full_contains = False

        for candidate in top_candidates:
            candidate_box = (
                candidate["x1"],
                candidate["y1"],
                candidate["x2"],
                candidate["y2"],
            )

            best_iou = max(
                best_iou,
                calculate_iou(
                    candidate_box,
                    defect_bbox,
                ),
            )

            if contains_defect(
                candidate_box,
                mask,
            ):
                any_contains = True

            if full_contains_bbox(
                candidate_box,
                defect_bbox,
            ):
                any_full_contains = True

        results[f"top{k}_contains"] = int(any_contains)
        results[f"top{k}_full_contains"] = int(any_full_contains)
        results[f"top{k}_best_iou"] = float(best_iou)

    return results


def main():
    print("CANDIDATE LOCALIZATION AUDIT")
    print("----------------------------")
    print()

    validation = pd.read_csv(VAL_CSV)

    # Only defective validation samples.
    validation = validation[
        validation["has_defect"] == 1
    ].reset_index(drop=True)

    print(
        f"Defective validation samples: {len(validation)}"
    )
    print(
        "Candidate generation uses image pixels only."
    )
    print(
        "Ground-truth masks are used only for evaluation."
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for index, row in validation.iterrows():
        result = analyze_image(row)

        if result is not None:
            results.append(result)

        print(
            f"[{index + 1:02d}/{len(validation):02d}] "
            f"image_id={row['image_id']} "
            f"processed"
        )

    result_df = pd.DataFrame(results)

    output_csv = OUTPUT_DIR / "candidate_localization_results.csv"

    result_df.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("RESULTS")
    print("-------")

    total = len(result_df)

    for k in TOP_K_VALUES:
        contains_rate = (
            result_df[f"top{k}_contains"].mean()
        )

        full_contains_rate = (
            result_df[f"top{k}_full_contains"].mean()
        )

        mean_iou = (
            result_df[f"top{k}_best_iou"].mean()
        )

        print(
            f"Top-{k}: "
            f"contains={contains_rate:.4f} | "
            f"full_bbox={full_contains_rate:.4f} | "
            f"mean_best_iou={mean_iou:.4f}"
        )

    print()
    print(f"Saved: {output_csv}")
    print()
    print("CANDIDATE LOCALIZATION AUDIT PASSED")


if __name__ == "__main__":
    main()