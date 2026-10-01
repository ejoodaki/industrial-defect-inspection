import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import KolektorDataset


TRAIN_CSV = Path("splits/train.csv")

TARGET_SIZE = 384
BATCH_SIZE = 2
NUM_EPOCHS_TO_SIMULATE = 10


def main():
    print("L-01 BALANCED SAMPLER AUDIT")
    print("===========================")

    dataset = KolektorDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE,
    )

    data = pd.read_csv(TRAIN_CSV)

    class_counts = data["has_defect"].value_counts().to_dict()

    healthy_count = class_counts[0]
    defective_count = class_counts[1]

    print(f"Dataset samples:    {len(dataset)}")
    print(f"Healthy samples:    {healthy_count}")
    print(f"Defective samples:  {defective_count}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Simulated epochs:   {NUM_EPOCHS_TO_SIMULATE}")
    print()

    weights = []

    for label in data["has_defect"]:
        if label == 0:
            weights.append(1.0 / healthy_count)
        else:
            weights.append(1.0 / defective_count)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )

    total_batches = 0
    all_healthy_batches = 0
    mixed_batches = 0
    all_defective_batches = 0

    total_healthy_samples = 0
    total_defective_samples = 0

    for epoch in range(NUM_EPOCHS_TO_SIMULATE):

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
        )

        epoch_all_healthy = 0
        epoch_mixed = 0
        epoch_all_defective = 0

        epoch_healthy_samples = 0
        epoch_defective_samples = 0

        for batch_indices in loader.batch_sampler:

            batch_indices = list(batch_indices)

            defective_count_in_batch = sum(
                data.iloc[index]["has_defect"] == 1
                for index in batch_indices
            )

            healthy_count_in_batch = (
                len(batch_indices)
                - defective_count_in_batch
            )

            total_batches += 1

            epoch_healthy_samples += healthy_count_in_batch
            epoch_defective_samples += defective_count_in_batch

            total_healthy_samples += healthy_count_in_batch
            total_defective_samples += defective_count_in_batch

            if defective_count_in_batch == 0:
                all_healthy_batches += 1
                epoch_all_healthy += 1

            elif defective_count_in_batch == len(batch_indices):
                all_defective_batches += 1
                epoch_all_defective += 1

            else:
                mixed_batches += 1
                epoch_mixed += 1

        print(
            f"Epoch {epoch + 1:2d}: "
            f"all-healthy={epoch_all_healthy:3d}, "
            f"mixed={epoch_mixed:3d}, "
            f"all-defective={epoch_all_defective:3d}, "
            f"healthy={epoch_healthy_samples:3d}, "
            f"defective={epoch_defective_samples:3d}"
        )

    print()
    print("OVERALL RESULTS")
    print("---------------")

    print(
        f"Total batches:          {total_batches}"
    )

    print(
        f"All-healthy batches:    {all_healthy_batches}"
    )

    print(
        f"Mixed batches:          {mixed_batches}"
    )

    print(
        f"All-defective batches:  {all_defective_batches}"
    )

    print()

    print(
        f"All-healthy batch rate: "
        f"{all_healthy_batches / total_batches:.4f}"
    )

    print(
        f"Mixed batch rate:       "
        f"{mixed_batches / total_batches:.4f}"
    )

    print(
        f"All-defective rate:     "
        f"{all_defective_batches / total_batches:.4f}"
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

    print("BALANCED SAMPLER AUDIT COMPLETED")


if __name__ == "__main__":
    main()