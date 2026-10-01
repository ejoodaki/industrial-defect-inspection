import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import KolektorDataset


TRAIN_CSV = Path("splits/train.csv")

TARGET_SIZE = 384
BATCH_SIZE = 2
NUM_EPOCHS_TO_SIMULATE = 10


def main():
    print("L-01 TRAINING BATCH COMPOSITION AUDIT")
    print("=====================================")

    dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    data = pd.read_csv(TRAIN_CSV)

    defective_indices = set(
        data.index[
            data["has_defect"] == 1
        ].tolist()
    )

    healthy_indices = set(
        data.index[
            data["has_defect"] == 0
        ].tolist()
    )

    print(f"Dataset samples:    {len(dataset)}")
    print(f"Healthy samples:    {len(healthy_indices)}")
    print(f"Defective samples:  {len(defective_indices)}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Simulated epochs:   {NUM_EPOCHS_TO_SIMULATE}")
    print()

    total_batches = 0
    all_healthy_batches = 0
    mixed_batches = 0
    all_defective_batches = 0

    total_defective_samples = 0
    total_healthy_samples = 0

    for epoch in range(NUM_EPOCHS_TO_SIMULATE):

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        epoch_all_healthy = 0
        epoch_mixed = 0
        epoch_all_defective = 0

        for batch_indices in loader.batch_sampler:

            batch_indices = list(batch_indices)

            defective_count = sum(
                index in defective_indices
                for index in batch_indices
            )

            healthy_count = len(batch_indices) - defective_count

            total_batches += 1

            total_defective_samples += defective_count
            total_healthy_samples += healthy_count

            if defective_count == 0:
                all_healthy_batches += 1
                epoch_all_healthy += 1

            elif defective_count == len(batch_indices):
                all_defective_batches += 1
                epoch_all_defective += 1

            else:
                mixed_batches += 1
                epoch_mixed += 1

        print(
            f"Epoch {epoch + 1:2d}: "
            f"all-healthy={epoch_all_healthy:3d}, "
            f"mixed={epoch_mixed:3d}, "
            f"all-defective={epoch_all_defective:3d}"
        )

    print()
    print("OVERALL RESULTS")
    print("---------------")

    print(
        f"Total batches:          {total_batches}"
    )

    print(
        f"All-healthy batches:    "
        f"{all_healthy_batches}"
    )

    print(
        f"Mixed batches:          "
        f"{mixed_batches}"
    )

    print(
        f"All-defective batches:  "
        f"{all_defective_batches}"
    )

    print()

    print(
        f"All-healthy batch rate:  "
        f"{all_healthy_batches / total_batches:.4f}"
    )

    print(
        f"Mixed batch rate:        "
        f"{mixed_batches / total_batches:.4f}"
    )

    print(
        f"All-defective batch rate:"
        f" {all_defective_batches / total_batches:.4f}"
    )

    print()

    print(
        f"Observed healthy samples:   "
        f"{total_healthy_samples}"
    )

    print(
        f"Observed defective samples: "
        f"{total_defective_samples}"
    )

    print()

    print("TRAINING BATCH AUDIT COMPLETED")


if __name__ == "__main__":
    main()