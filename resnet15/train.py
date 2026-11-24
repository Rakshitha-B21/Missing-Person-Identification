"""Train ResNet-50 on the QMUL-SurvFace training split using CPU PyTorch."""

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

if __package__:
    from .dataset import QMULSurvFaceDataset
    from .model import ResNet50
    from .test_checkpoint import load_checkpoint, validate_checkpoint_architecture
else:
    from dataset import QMULSurvFaceDataset
    from model import ResNet50
    from test_checkpoint import load_checkpoint, validate_checkpoint_architecture


EXPECTED_IDENTITIES = 5319
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32


def positive_int(value: str) -> int:
    """Parse a positive integer for command-line options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for command-line options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def minimum_two_int(value: str) -> int:
    """Parse an integer of at least two for options that need both splits."""
    parsed = positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError(
            f"expected an integer of at least 2, got {value!r}"
        )
    return parsed


def select_identity_folders(
    class_to_idx: Dict[str, int], max_identities: Optional[int], seed: int
) -> List[Tuple[str, int]]:
    """Select identity folders reproducibly, retaining their original labels."""
    identities = sorted(class_to_idx.items(), key=lambda item: item[1])
    if max_identities is None or max_identities == len(identities):
        return identities
    if max_identities > len(identities):
        raise ValueError(
            f"--max-identities is {max_identities}, but only "
            f"{len(identities)} identity folders are available."
        )
    if max_identities < 1:
        raise ValueError("--max-identities must be at least 1.")

    selected = random.Random(seed).sample(identities, max_identities)
    return sorted(selected, key=lambda item: item[1])


def select_images_per_identity(
    labels: Sequence[int],
    max_images_per_identity: Optional[int],
    seed: int,
    max_total_images: Optional[int] = None,
) -> List[int]:
    """Select a seeded subset while preserving dataset order and split eligibility."""
    indices_by_identity: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_identity[int(label)].append(index)

    if max_total_images is None and max_images_per_identity is None:
        return list(range(len(labels)))

    rng = random.Random(seed)
    candidates_by_identity: Dict[int, List[int]] = {}
    for label in sorted(indices_by_identity):
        identity_indices = indices_by_identity[label]
        if (
            max_images_per_identity is not None
            and len(identity_indices) > max_images_per_identity
        ):
            identity_indices = rng.sample(identity_indices, max_images_per_identity)
        candidates_by_identity[label] = identity_indices

    if max_total_images is None:
        selected_indices = {
            index
            for identity_indices in candidates_by_identity.values()
            for index in identity_indices
        }
        return [index for index in range(len(labels)) if index in selected_indices]

    minimum_total = 2 * len(candidates_by_identity)
    if max_total_images < minimum_total:
        raise ValueError(
            f"--max-total-images must be at least {minimum_total} to keep "
            "two images per identity."
        )

    for label, identity_indices in candidates_by_identity.items():
        if len(identity_indices) < 2:
            raise ValueError(
                f"Identity label {label} has fewer than two images; "
                "cannot include it in both train and validation."
            )

    selected_indices = set()
    remaining_indices = []
    for identity_indices in candidates_by_identity.values():
        selected_indices.update(rng.sample(identity_indices, 2))
        remaining_indices.extend(
            index for index in identity_indices if index not in selected_indices
        )

    additional_count = min(max_total_images - minimum_total, len(remaining_indices))
    selected_indices.update(rng.sample(remaining_indices, additional_count))

    return [index for index in range(len(labels)) if index in selected_indices]


class RemappedLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, label_mapping: Dict[int, int]) -> None:
        self.dataset = dataset
        self.label_mapping = label_mapping

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, self.label_mapping[int(label)]

    def __len__(self) -> int:
        return len(self.dataset)


def split_by_identity(
    labels: Sequence[int], seed: int, train_fraction: float = 0.8
) -> Tuple[List[int], List[int]]:
    """Create disjoint train/validation indices, splitting each identity."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    indices_by_identity: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_identity[int(label)].append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices: List[int] = []
    validation_indices: List[int] = []

    for label in sorted(indices_by_identity):
        identity_indices = indices_by_identity[label]
        if len(identity_indices) < 2:
            raise ValueError(
                f"Identity label {label} has fewer than two images; "
                "cannot include it in both train and validation."
            )

        shuffled = torch.randperm(len(identity_indices), generator=generator).tolist()
        train_count = int(len(identity_indices) * train_fraction)
        train_count = max(1, min(len(identity_indices) - 1, train_count))
        train_indices.extend(identity_indices[position] for position in shuffled[:train_count])
        validation_indices.extend(
            identity_indices[position] for position in shuffled[train_count:]
        )

    return train_indices, validation_indices


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader)
    progress_interval = max(1, total_batches // 10)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to("cpu")
        labels = labels.to("cpu", dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        _, logits = model(images)
        if logits is None:
            raise RuntimeError("ResNet-50 was created without an identity classification head.")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_index % progress_interval == 0 or batch_index == total_batches:
            print(
                f"Epoch {epoch}/{epochs} - "
                f"training batches {batch_index}/{total_batches}, "
                f"running loss {total_loss / total_samples:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("Training split is empty; no training batches were produced.")
    return total_loss / total_samples


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to("cpu")
            labels = labels.to("cpu", dtype=torch.long)
            _, logits = model(images)
            if logits is None:
                raise RuntimeError(
                    "ResNet-50 was created without an identity classification head."
                )

            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation split is empty; no validation batches were produced.")
    return total_loss / total_samples, total_correct / total_samples


def run_training(
    epochs: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    max_images_per_identity: Optional[int] = None,
    max_total_images: Optional[int] = None,
    max_identities: Optional[int] = None,
    resume: bool = False,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    print("Loading QMUL-SurvFace training split...", flush=True)
    dataset = QMULSurvFaceDataset()
    identity_count = len(dataset.class_to_idx)
    if identity_count != EXPECTED_IDENTITIES:
        raise ValueError(
            f"Expected {EXPECTED_IDENTITIES} training identities, found {identity_count} "
            f"under {dataset.root}."
        )
    selected_identity_folders = select_identity_folders(
        dataset.class_to_idx, max_identities, seed
    )
    identity_to_label = {
        identity: selected_label
        for selected_label, (identity, _) in enumerate(selected_identity_folders)
    }
    original_to_selected_label = {
        original_label: identity_to_label[identity]
        for identity, original_label in selected_identity_folders
    }

    selected_source_indices = []
    selected_original_labels = []
    for index, (_, label) in enumerate(dataset.samples):
        if label in original_to_selected_label:
            selected_source_indices.append(index)
            selected_original_labels.append(label)

    image_counts: Dict[int, int] = defaultdict(int)
    for label in selected_original_labels:
        image_counts[label] += 1
    for identity, original_label in selected_identity_folders:
        if image_counts[original_label] < 2:
            raise ValueError(
                f"Selected identity folder '{identity}' has fewer than two images; "
                "cannot include it in both train and validation."
            )

    selected_indices = select_images_per_identity(
        selected_original_labels,
        max_images_per_identity,
        seed,
        max_total_images,
    )
    print(
        f"Selected {len(identity_to_label):,} identities and "
        f"{len(selected_indices):,} selected images.",
        flush=True,
    )
    selected_source_indices = [
        selected_source_indices[index] for index in selected_indices
    ]
    selected_labels = [
        original_to_selected_label[selected_original_labels[index]]
        for index in selected_indices
    ]
    train_split_indices, validation_split_indices = split_by_identity(
        selected_labels, seed=seed
    )
    train_indices = [
        selected_source_indices[index] for index in train_split_indices
    ]
    validation_indices = [
        selected_source_indices[index] for index in validation_split_indices
    ]
    remapped_dataset = RemappedLabelDataset(dataset, original_to_selected_label)
    train_set = Subset(remapped_dataset, train_indices)
    validation_set = Subset(remapped_dataset, validation_indices)
    print(
        f"Split: {len(train_set):,} training images, "
        f"{len(validation_set):,} validation images.",
        flush=True,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(
    train_set,
    shuffle=True,
    drop_last=True,
    **loader_options
)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "resnet50_best_model.pth"
    )
    model = ResNet50(
        num_classes=len(identity_to_label), pretrained=not resume
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_architecture(checkpoint)
        checkpoint_identity_mapping = checkpoint.get("identity_to_label")
        if checkpoint_identity_mapping != identity_to_label:
            raise ValueError(
                "The ResNet-50 checkpoint identity mapping does not match the "
                "current training selection; resume with the same identities."
            )
        model_state = checkpoint.get("model_state_dict")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        checkpoint_epoch = checkpoint.get("epoch")
        if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
            raise ValueError(
                "ResNet-50 checkpoint is missing model or optimizer state."
            )
        if (
            not isinstance(checkpoint_epoch, int)
            or isinstance(checkpoint_epoch, bool)
            or checkpoint_epoch < 1
        ):
            raise ValueError("ResNet-50 checkpoint has an invalid epoch.")
        try:
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not resume from ResNet-50 checkpoint "
                f"'{checkpoint_path}': {exc}"
            ) from exc
        checkpoint_validation_loss = checkpoint.get("validation_loss")
        if (
            not isinstance(checkpoint_validation_loss, (int, float))
            or isinstance(checkpoint_validation_loss, bool)
            or not math.isfinite(checkpoint_validation_loss)
        ):
            raise ValueError("ResNet-50 checkpoint has an invalid validation loss.")
        best_validation_loss = float(checkpoint_validation_loss)
        start_epoch = checkpoint_epoch + 1
        print(
            f"Resuming ResNet-50 training from epoch {checkpoint_epoch}; "
            f"best validation loss: {best_validation_loss:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}", flush=True)
        training_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, epochs
        )
        validation_loss, validation_accuracy = validate(model, validation_loader, criterion)

        print(
            f"Epoch {epoch}/{epochs} summary - "
            f"training loss: {training_loss:.4f}, "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.2%}",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            checkpoint = {
                "architecture": ResNet50.architecture,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "identity_to_label": identity_to_label,
            }
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not save checkpoint to '{checkpoint_path}': {exc}"
                ) from exc

            best_validation_loss = validation_loss
            print(f"Saved best checkpoint to {checkpoint_path}", flush=True)

    print("Training complete.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=DEFAULT_EPOCHS,
        help=f"number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"batch size for training and validation (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=0,
        help="data-loader worker processes (default: 0; Windows-safe)",
    )
    parser.add_argument(
        "--max-images-per-identity",
        type=minimum_two_int,
        default=None,
        help="maximum images per identity (default: all images; minimum: 2)",
    )
    parser.add_argument(
        "--max-total-images",
        type=positive_int,
        default=None,
        help="maximum total images to select (default: all available images)",
    )
    parser.add_argument(
        "--max-identities",
        type=positive_int,
        default=None,
        help=f"maximum identity folders to select (default: all {EXPECTED_IDENTITIES})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from checkpoints/resnet50_best_model.pth",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_training(
            args.epochs,
            args.batch_size,
            args.seed,
            args.num_workers,
            args.max_images_per_identity,
            args.max_total_images,
            args.max_identities,
            args.resume,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr, flush=True)
        return 130
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
