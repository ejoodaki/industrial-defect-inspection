import cv2
import numpy as np
import pandas as pd

from audit_candidate_localization import (
    build_candidate_score,
    bbox_from_mask,
    calculate_iou,
    contains_defect,
    full_contains_bbox,
    load_gray,
)


VAL_CSV = "splits/validation.csv"

PATCH_SIZES = [128, 192, 256]
STRIDE = 32
TOP_K = 5


def generate_multiscale_candidates(score, patch_sizes, stride):
    height, width = score.shape
    candidates = []

    for patch_size in patch_sizes:
        if patch_size > height or patch_size > width:
            continue

        for y1 in range(0, height - patch_size + 1, stride):
            for x1 in range(0, width - patch_size + 1, stride):
                y2 = y1 + patch_size
                x2 = x1 + patch_size

                patch_score = score[y1:y2, x1:x2]

                candidates.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "patch_size": patch_size,
                        "score": float(np.mean(patch_score)),
                    }
                )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates


def evaluate_top_k(candidates, defect_bbox, mask, top_k):
    selected = candidates[:top_k]

    contains_count = 0
    full_bbox_count = 0
    best_iou = 0.0

    for candidate in selected:
        bbox = (
            candidate["x1"],
            candidate["y1"],
            candidate["x2"],
            candidate["y2"],
        )

        candidate_bbox = (
            candidate["x1"],
            candidate["y1"],
            candidate["x2"],
            candidate["y2"],
        )

        if contains_defect(candidate_bbox, mask):
            contains_count = 1

        if full_contains_bbox(candidate_bbox, defect_bbox):
            full_bbox_count = 1

        iou = calculate_iou(defect_bbox, candidate_bbox)
        best_iou = max(best_iou, iou)

    return contains_count, full_bbox_count, best_iou


def main():
    print("MULTI-SCALE CANDIDATE LOCALIZATION AUDIT")
    print("-----------------------------------------")
    print()

    validation = pd.read_csv(VAL_CSV)
    validation = validation[
        validation["has_defect"] == 1
    ].reset_index(drop=True)

    print(f"Defective validation samples: {len(validation)}")
    print(f"Patch sizes: {PATCH_SIZES}")
    print(f"Stride: {STRIDE}")
    print(f"Top-K: {TOP_K}")
    print()

    results = []

    for index, row in validation.iterrows():
        image_id = int(row["image_id"])

        image = load_gray(row["image_path"])
        mask = load_gray(row["mask_path"])

        defect_bbox = bbox_from_mask(mask)

        score = build_candidate_score(image)

        candidates = generate_multiscale_candidates(
            score,
            PATCH_SIZES,
            STRIDE,
        )

        contains, full_bbox, best_iou = evaluate_top_k(
            candidates,
            defect_bbox,
            mask,
            TOP_K,
        )

        results.append(
            {
                "image_id": image_id,
                "contains": contains,
                "full_bbox": full_bbox,
                "best_iou": best_iou,
            }
        )

        print(
            f"[{index + 1:02d}/{len(validation)}] "
            f"image_id={image_id} | "
            f"contains={contains} | "
            f"full_bbox={full_bbox} | "
            f"best_iou={best_iou:.4f}"
        )

    results_df = pd.DataFrame(results)

    contains_rate = results_df["contains"].mean()
    full_bbox_rate = results_df["full_bbox"].mean()
    mean_iou = results_df["best_iou"].mean()

    print()
    print("RESULTS")
    print("-------")
    print(
        f"Multi-scale Top-5 contains: "
        f"{contains_rate:.4f}"
    )
    print(
        f"Multi-scale Top-5 full bbox: "
        f"{full_bbox_rate:.4f}"
    )
    print(
        f"Multi-scale Top-5 mean best IoU: "
        f"{mean_iou:.4f}"
    )

    output_path = (
        "outputs/candidate_localization/"
        "multiscale_localization_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path}")
    print()
    print("MULTI-SCALE LOCALIZATION AUDIT PASSED")


if __name__ == "__main__":
    main()