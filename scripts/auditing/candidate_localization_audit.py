from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

VAL_CSV = ROOT / "splits" / "validation.csv"

PATCH_SIZE = 192
TOP_K = 5


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Failed to read: {path}")

    return image


def generate_windows(height, width, patch_size):
    xs = [0]

    if width > patch_size:
        xs.append(width - patch_size)

    ys = list(
        range(
            0,
            max(1, height - patch_size + 1),
            max(1, patch_size // 4),
        )
    )

    if height > patch_size:
        last_y = height - patch_size
        if last_y not in ys:
            ys.append(last_y)

    windows = []

    for y in ys:
        for x in xs:
            windows.append((x, y))

    return windows


def score_patch(image, x, y):
    patch = image[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE,
    ]

    # Local high-frequency response.
    blurred = cv2.GaussianBlur(
        patch,
        (0, 0),
        sigmaX=5,
    )

    residual = cv2.absdiff(
        patch,
        blurred,
    )

    # Gradient magnitude.
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

    gradient = cv2.magnitude(gx, gy)

    # Combine both signals.
    score = (
        float(residual.mean())
        + 0.25 * float(gradient.mean())
    )

    return score


def bbox_iou(candidate, gt_bbox):
    cx, cy, cw, ch = candidate
    gx, gy, gw, gh = gt_bbox

    c_x2 = cx + cw
    c_y2 = cy + ch

    g_x2 = gx + gw
    g_y2 = gy + gh

    ix1 = max(cx, gx)
    iy1 = max(cy, gy)
    ix2 = min(c_x2, g_x2)
    iy2 = min(c_y2, g_y2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    candidate_area = cw * ch
    gt_area = gw * gh

    union = candidate_area + gt_area - intersection

    if union == 0:
        return 0.0

    return intersection / union


def contains_defect(candidate, mask):
    x, y, _, _ = candidate

    crop = mask[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE,
    ]

    return bool(np.any(crop > 0))


def main():

    print("CANDIDATE LOCALIZATION AUDIT")
    print("============================")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Top-K candidates: {TOP_K}")
    print()

    df = pd.read_csv(VAL_CSV)

    defective = df[
        df["has_defect"] == 1
    ].copy()

    print(
        f"Validation defective images: {len(defective)}"
    )
    print()

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    top1_full = 0
    top3_full = 0
    top5_full = 0

    top1_ious = []
    top5_ious = []

    for _, row in defective.iterrows():

        image_path = ROOT / row["image_path"]
        mask_path = ROOT / row["mask_path"]

        image = load_gray(image_path)
        mask = load_gray(mask_path)

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        gx1 = int(xs.min())
        gy1 = int(ys.min())
        gx2 = int(xs.max())
        gy2 = int(ys.max())

        gt_bbox = (
            gx1,
            gy1,
            gx2 - gx1 + 1,
            gy2 - gy1 + 1,
        )

        candidates = []

        for x, y in generate_windows(
            image.shape[0],
            image.shape[1],
            PATCH_SIZE,
        ):

            score = score_patch(
                image,
                x,
                y,
            )

            candidate = (
                x,
                y,
                PATCH_SIZE,
                PATCH_SIZE,
            )

            candidates.append(
                (score, candidate)
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        top_candidates = [
            candidate
            for _, candidate in candidates[:TOP_K]
        ]

        # Candidate contains any GT defect pixel.
        hits = [
            contains_defect(
                candidate,
                mask,
            )
            for candidate in top_candidates
        ]

        if hits[0]:
            top1_hits += 1

        if any(hits[:3]):
            top3_hits += 1

        if any(hits[:5]):
            top5_hits += 1

        # Full-defect containment.
        def fully_contains(candidate):
            x, y, _, _ = candidate

            return (
                x <= gx1
                and y <= gy1
                and x + PATCH_SIZE > gx2
                and y + PATCH_SIZE > gy2
            )

        full_hits = [
            fully_contains(candidate)
            for candidate in top_candidates
        ]

        if full_hits[0]:
            top1_full += 1

        if any(full_hits[:3]):
            top3_full += 1

        if any(full_hits[:5]):
            top5_full += 1

        top1_ious.append(
            bbox_iou(
                top_candidates[0],
                gt_bbox,
            )
        )

        top5_ious.append(
            max(
                bbox_iou(
                    candidate,
                    gt_bbox,
                )
                for candidate in top_candidates
            )
        )

    total = len(defective)

    print("DEFECT COVERAGE")
    print("---------------")
    print(
        f"Top-1 contains defect: "
        f"{top1_hits}/{total} "
        f"({top1_hits / total:.2%})"
    )
    print(
        f"Top-3 contains defect: "
        f"{top3_hits}/{total} "
        f"({top3_hits / total:.2%})"
    )
    print(
        f"Top-5 contains defect: "
        f"{top5_hits}/{total} "
        f"({top5_hits / total:.2%})"
    )

    print()
    print("FULL DEFECT CONTAINMENT")
    print("-----------------------")
    print(
        f"Top-1 full containment: "
        f"{top1_full}/{total} "
        f"({top1_full / total:.2%})"
    )
    print(
        f"Top-3 full containment: "
        f"{top3_full}/{total} "
        f"({top3_full / total:.2%})"
    )
    print(
        f"Top-5 full containment: "
        f"{top5_full}/{total} "
        f"({top5_full / total:.2%})"
    )

    print()
    print("BBOX IoU")
    print("--------")
    print(
        f"Top-1 mean IoU: "
        f"{np.mean(top1_ious):.4f}"
    )
    print(
        f"Top-5 best mean IoU: "
        f"{np.mean(top5_ious):.4f}"
    )

    print()
    print("AUDIT COMPLETED")


if __name__ == "__main__":
    main()