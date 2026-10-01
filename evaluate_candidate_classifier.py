from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from audit_candidate_localization import (
    build_candidate_score,
    generate_candidates,
    bbox_from_mask,
    calculate_iou,
    contains_defect,
    full_contains_bbox,
)

from patch_classifier import PatchClassifier


ROOT = Path(".")
VAL_CSV = ROOT / "splits" / "validation.csv"

CHECKPOINT = (
    ROOT
    / "outputs"
    / "patch_classifier"
    / "classifier_short_best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "candidate_localization"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "candidate_classifier_results.csv"
)

PATCH_SIZE = 192
TOP_K = 5


def load_gray(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read: {path}"
        )

    return image


def load_classifier():
    device = torch.device("cpu")

    model = PatchClassifier().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def classify_patch(model, device, patch):
    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"Unexpected patch shape: {patch.shape}"
        )

    patch_tensor = (
        torch.from_numpy(
            patch.astype(np.float32) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logit = model(patch_tensor)
        probability = torch.sigmoid(logit)

    return float(probability.item())


def evaluate_candidate(candidate, mask, defect_bbox):
    box = (
        candidate["x1"],
        candidate["y1"],
        candidate["x2"],
        candidate["y2"],
    )

    return {
        "contains": int(
            contains_defect(box, mask)
        ),
        "full_contains": int(
            full_contains_bbox(
                box,
                defect_bbox,
            )
        ),
        "iou": float(
            calculate_iou(
                box,
                defect_bbox,
            )
        ),
    }


def main():
    print("CANDIDATE + CLASSIFIER EVALUATION")
    print("---------------------------------")
    print()

    validation = pd.read_csv(VAL_CSV)

    validation = validation[
        validation["has_defect"] == 1
    ].reset_index(drop=True)

    print(
        f"Defective validation samples: {len(validation)}"
    )
    print(
        f"Candidate pool: Top-{TOP_K}"
    )
    print()

    model, device = load_classifier()

    print(
        f"Classifier checkpoint: {CHECKPOINT}"
    )
    print(
        f"Device: {device}"
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for index, row in validation.iterrows():
        image_id = int(row["image_id"])

        image = load_gray(
            ROOT / row["image_path"]
        )

        mask = load_gray(
            ROOT / row["mask_path"]
        )

        defect_bbox = bbox_from_mask(mask)

        if defect_bbox is None:
            continue

        score = build_candidate_score(image)

        candidates = generate_candidates(score)

        top_candidates = candidates[:TOP_K]

        if len(top_candidates) < TOP_K:
            raise ValueError(
                f"Not enough candidates for image {image_id}"
            )

        scored_candidates = []

        for rank, candidate in enumerate(
            top_candidates,
            start=1,
        ):
            x1 = candidate["x1"]
            y1 = candidate["y1"]
            x2 = candidate["x2"]
            y2 = candidate["y2"]

            patch = image[
                y1:y2,
                x1:x2,
            ]

            probability = classify_patch(
                model,
                device,
                patch,
            )

            geometry = evaluate_candidate(
                candidate,
                mask,
                defect_bbox,
            )

            scored_candidates.append(
                {
                    "original_rank": rank,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "candidate_score": candidate["score"],
                    "classifier_probability": probability,
                    **geometry,
                }
            )

        # Candidate chosen by classifier.
        selected = max(
            scored_candidates,
            key=lambda item: item[
                "classifier_probability"
            ],
        )

        # Best candidate according to GT.
        # Used ONLY for analysis, never for selection.
        best_gt = max(
            scored_candidates,
            key=lambda item: item["iou"],
        )

        # Classical Top-1.
        classical_top1 = scored_candidates[0]

        result = {
            "image_id": image_id,

            "classical_top1_contains":
                classical_top1["contains"],

            "classical_top1_full_contains":
                classical_top1["full_contains"],

            "classical_top1_iou":
                classical_top1["iou"],

            "top5_contains":
                int(
                    any(
                        item["contains"]
                        for item in scored_candidates
                    )
                ),

            "top5_full_contains":
                int(
                    any(
                        item["full_contains"]
                        for item in scored_candidates
                    )
                ),

            "top5_best_iou":
                max(
                    item["iou"]
                    for item in scored_candidates
                ),

            "classifier_selected_original_rank":
                selected["original_rank"],

            "classifier_selected_probability":
                selected["classifier_probability"],

            "classifier_selected_contains":
                selected["contains"],

            "classifier_selected_full_contains":
                selected["full_contains"],

            "classifier_selected_iou":
                selected["iou"],

            "gt_best_rank":
                best_gt["original_rank"],

            "gt_best_iou":
                best_gt["iou"],
        }

        results.append(result)

        print(
            f"[{index + 1:02d}/{len(validation):02d}] "
            f"image_id={image_id} | "
            f"classifier_rank="
            f"{selected['original_rank']} | "
            f"prob="
            f"{selected['classifier_probability']:.4f} | "
            f"IoU="
            f"{selected['iou']:.4f}"
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("RESULTS")
    print("-------")

    print(
        "Classical Top-1:"
    )
    print(
        f"  contains: "
        f"{result_df['classical_top1_contains'].mean():.4f}"
    )
    print(
        f"  full bbox: "
        f"{result_df['classical_top1_full_contains'].mean():.4f}"
    )
    print(
        f"  mean IoU: "
        f"{result_df['classical_top1_iou'].mean():.4f}"
    )

    print()

    print(
        "Classical Top-5:"
    )
    print(
        f"  contains: "
        f"{result_df['top5_contains'].mean():.4f}"
    )
    print(
        f"  full bbox: "
        f"{result_df['top5_full_contains'].mean():.4f}"
    )
    print(
        f"  mean best IoU: "
        f"{result_df['top5_best_iou'].mean():.4f}"
    )

    print()

    print(
        "Classifier-selected candidate:"
    )
    print(
        f"  contains: "
        f"{result_df['classifier_selected_contains'].mean():.4f}"
    )
    print(
        f"  full bbox: "
        f"{result_df['classifier_selected_full_contains'].mean():.4f}"
    )
    print(
        f"  mean IoU: "
        f"{result_df['classifier_selected_iou'].mean():.4f}"
    )

    print()

    print(
        "CLASSIFIER RANK SELECTION"
    )
    print(
        "-------------------------"
    )

    rank_counts = (
        result_df[
            "classifier_selected_original_rank"
        ]
        .value_counts()
        .sort_index()
    )

    for rank, count in rank_counts.items():
        print(
            f"Top-{rank}: "
            f"{count}/{len(result_df)}"
        )

    print()

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print(
        "CANDIDATE + CLASSIFIER "
        "EVALUATION PASSED"
    )


if __name__ == "__main__":
    main()