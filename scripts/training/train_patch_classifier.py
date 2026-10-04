from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from patch_classifier import PatchClassifier
from patch_classifier_dataset import PatchClassifierDataset


TRAIN_CSVS = [
    "outputs/patch_dataset/positive_train.csv",
    "outputs/patch_dataset/hard_negative_train.csv",
    "outputs/patch_dataset/healthy_negative_train.csv",
]

VALIDATION_CSVS = [
    "outputs/patch_dataset/positive_validation.csv",
    "outputs/patch_dataset/hard_negative_validation.csv",
    "outputs/patch_dataset/healthy_negative_validation.csv",
]

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001


def get_labels(dataset):
    return [
        int(sample["label"])
        for sample in dataset.samples
    ]


def build_balanced_sampler(dataset):
    labels = get_labels(dataset)

    positive_count = sum(
        label == 1 for label in labels
    )
    negative_count = sum(
        label == 0 for label in labels
    )

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Both positive and negative samples are required."
        )

    class_weights = {
        0: 1.0 / negative_count,
        1: 1.0 / positive_count,
    }

    sample_weights = torch.tensor(
        [
            class_weights[label]
            for label in labels
        ],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += (
                loss.item() * labels.size(0)
            )
            total += labels.size(0)

            probabilities = torch.sigmoid(logits)
            predictions = (
                probabilities >= 0.5
            ).float()

            tp += int(
                ((predictions == 1) & (labels == 1))
                .sum()
                .item()
            )

            tn += int(
                ((predictions == 0) & (labels == 0))
                .sum()
                .item()
            )

            fp += int(
                ((predictions == 1) & (labels == 0))
                .sum()
                .item()
            )

            fn += int(
                ((predictions == 0) & (labels == 1))
                .sum()
                .item()
            )

    avg_loss = total_loss / total

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if fn + tp > 0
        else 0.0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    device = torch.device("cpu")

    print("PATCH CLASSIFIER TRAINING")
    print("-------------------------")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    train_dataset = PatchClassifierDataset(
        TRAIN_CSVS
    )

    validation_dataset = PatchClassifierDataset(
        VALIDATION_CSVS
    )

    train_sampler = build_balanced_sampler(
        train_dataset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = PatchClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(
        f"Train samples:       {len(train_dataset)}"
    )
    print(
        f"Validation samples:  {len(validation_dataset)}"
    )
    print()

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            running_loss / total_samples
        )

        metrics = evaluate(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"FPR={metrics['false_positive_rate']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    output_dir = Path("outputs/patch_classifier")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir / "classifier_short_best.pt"
    )

    if best_state is None:
        raise RuntimeError(
            "No model checkpoint was produced."
        )

    torch.save(
        {
            "model_state_dict": best_state,
            "best_f1": best_f1,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        checkpoint_path,
    )

    print()
    print(
        f"Best validation F1: {best_f1:.4f}"
    )
    print(
        f"Checkpoint saved: {checkpoint_path}"
    )
    print()
    print("PATCH CLASSIFIER TRAINING PASSED")


if __name__ == "__main__":
    main()