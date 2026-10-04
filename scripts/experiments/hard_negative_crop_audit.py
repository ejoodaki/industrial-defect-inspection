from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"

PATCH_SIZE = 192
SAMPLES_PER_IMAGE = 100
RANDOM_SEED = 42

# Maximum allowed defect pixels in a hard-negative crop.
# 0 means absolutely no defect pixels.
MAX_DEFECT_PIXELS = 0


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    return image


def find_hard_negatives(mask, rng):
    height, width = mask.shape

    if height < PATCH_SIZE or width < PATCH_SIZE:
        return []

    defect_pixels = mask > 0

    ys, xs = np.where(defect_pixels)

    if len(xs) == 0:
        return []

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    candidates = []

    for _ in range(SAMPLES_PER_IMAGE):
        x = rng.randint(0, width - PATCH_SIZE)
        y = rng.randint(0, height - PATCH_SIZE)

        crop_mask = defect_pixels[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ]

        defect_count = int(crop_mask.sum())

        if defect_count > MAX_DEFECT_PIXELS:
            continue

        # Distance from crop rectangle to defect bounding box.
        crop_x1 = x
        crop_x2 = x + PATCH_SIZE - 1
        crop_y1 = y
        crop_y2 = y + PATCH_SIZE - 1

        dx = max(
            x_min - crop_x2,
            crop_x1 - x_max,
            0,
        )

        dy = max(
            y_min - crop_y2,
            crop_y1 - y_max,
            0,
        )

        distance = float(np.sqrt(dx * dx + dy * dy))

        candidates.append(
            {
                "x": x,
                "y": y,
                "distance_to_bbox": distance,
                "defect_pixels": defect_count,
            }
        )

    return candidates


def main():
    print("HARD NEGATIVE CROP AUDIT")
    print("=========================")
    print(f"Patch size:          {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Samples per image:   {SAMPLES_PER_IMAGE}")
    print(f"Max defect pixels:   {MAX_DEFECT_PIXELS}")
    print()

    df = pd.read_csv(TRAIN_CSV)

    defective = df[df["has_defect"] == 1].copy()

    print(f"Defective train images: {len(defective)}")
    print()

    rng = random.Random(RANDOM_SEED)

    all_candidates = []
    images_with_candidates = 0
    candidate_counts = []

    for _, row in defective.iterrows():
        image_path = ROOT / row["image_path"]
        mask_path = ROOT / row["mask_path"]

        mask = load_gray(mask_path)

        candidates = find_hard_negatives(mask, rng)

        if candidates:
            images_with_candidates += 1

        candidate_counts.append(len(candidates))

        for candidate in candidates:
            candidate["image_id"] = Path(row["image_path"]).stem
            candidate["image_path"] = str(image_path)
            all_candidates.append(candidate)

    print("RESULTS")
    print("-------")
    print(
        f"Images with hard negatives: "
        f"{images_with_candidates}/{len(defective)}"
    )

    print(
        f"Total hard-negative candidates: "
        f"{len(all_candidates)}"
    )

    if not all_candidates:
        print()
        print("AUDIT FAILED: No valid hard-negative crops found.")
        return

    counts = np.array(candidate_counts)

    print(
        f"Candidates per defective image:"
        f" min={counts.min()}"
        f" median={np.median(counts):.0f}"
        f" mean={counts.mean():.2f}"
        f" max={counts.max()}"
    )

    distances = np.array(
        [item["distance_to_bbox"] for item in all_candidates]
    )

    print()
    print("DISTANCE TO DEFECT BOUNDING BOX")
    print("--------------------------------")
    print(f"Min:    {distances.min():.2f}")
    print(f"P25:    {np.percentile(distances, 25):.2f}")
    print(f"Median: {np.median(distances):.2f}")
    print(f"P75:    {np.percentile(distances, 75):.2f}")
    print(f"Max:    {distances.max():.2f}")

    # Show the closest candidates.
    all_candidates.sort(key=lambda x: x["distance_to_bbox"])

    print()
    print("CLOSEST HARD NEGATIVES")
    print("----------------------")

    for item in all_candidates[:10]:
        print(
            f"{item['image_id']:>6} | "
            f"x={item['x']:>3} y={item['y']:>3} | "
            f"distance={item['distance_to_bbox']:>7.2f}"
        )

    # Save candidates for the next stage only if the audit is useful.
    output_dir = ROOT / "outputs" / "hard_negative_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "hard_negative_candidates.csv"

    pd.DataFrame(all_candidates).to_csv(
        output_csv,
        index=False,
    )

    print()
    print(f"Saved: {output_csv}")

    print()
    print("AUDIT COMPLETED")


if __name__ == "__main__":
    main()