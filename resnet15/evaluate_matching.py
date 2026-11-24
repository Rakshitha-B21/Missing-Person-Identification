"""Exploratory cosine-similarity evaluation on sampled training images."""

import random
import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

if __package__:
    from .dataset import QMULSurvFaceDataset
    from .model import ResNet50
    from .test_checkpoint import (
        load_checkpoint,
        validate_checkpoint_architecture,
        validate_identity_mapping,
    )
else:
    from dataset import QMULSurvFaceDataset
    from model import ResNet50
    from test_checkpoint import (
        load_checkpoint,
        validate_checkpoint_architecture,
        validate_identity_mapping,
    )


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "checkpoints" / "resnet50_best_model.pth"
)
RANDOM_SEED = 42
MAX_IDENTITIES = 100
MAX_IMAGES_PER_IDENTITY = 5
EMBEDDING_BATCH_SIZE = 32
SIMILARITY_BATCH_SIZE = 4096


def select_training_images(
    dataset: QMULSurvFaceDataset, identity_to_label: Mapping
) -> Tuple[List[int], Dict[str, List[int]], int]:
    original_label_to_identity = {
        original_label: identity
        for identity, original_label in dataset.class_to_idx.items()
        if identity in identity_to_label
    }
    samples_by_identity: Dict[str, List[int]] = defaultdict(list)
    for index, (_, label) in enumerate(dataset.samples):
        identity = original_label_to_identity.get(label)
        if identity is not None:
            samples_by_identity[identity].append(index)

    eligible_identities = sorted(
        identity
        for identity in identity_to_label
        if len(samples_by_identity[identity]) >= 2
    )
    insufficient_count = len(identity_to_label) - len(eligible_identities)
    if len(eligible_identities) < 2:
        raise ValueError(
            "At least two checkpoint identities with two or more training images "
            "are required to create genuine and impostor pairs."
        )

    rng = random.Random(RANDOM_SEED)
    selected_identities = rng.sample(
        eligible_identities, min(MAX_IDENTITIES, len(eligible_identities))
    )
    selected_identities.sort()

    image_indices: List[int] = []
    embedding_indices_by_identity: Dict[str, List[int]] = {}
    for identity in selected_identities:
        candidates = samples_by_identity[identity]
        selected_images = sorted(
            rng.sample(candidates, min(MAX_IMAGES_PER_IDENTITY, len(candidates)))
        )
        start = len(image_indices)
        image_indices.extend(selected_images)
        embedding_indices_by_identity[identity] = list(
            range(start, len(image_indices))
        )

    return image_indices, embedding_indices_by_identity, insufficient_count


def create_pairs(
    embedding_indices_by_identity: Mapping[str, Sequence[int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    identities = list(embedding_indices_by_identity)
    genuine_pairs = [
        pair
        for identity in identities
        for pair in combinations(embedding_indices_by_identity[identity], 2)
    ]
    impostor_pairs = [
        (left_index, right_index)
        for left_position, left_identity in enumerate(identities)
        for right_identity in identities[left_position + 1 :]
        for left_index, right_index in product(
            embedding_indices_by_identity[left_identity],
            embedding_indices_by_identity[right_identity],
        )
    ]
    if not genuine_pairs or not impostor_pairs:
        raise ValueError(
            "The selected images did not produce both genuine and impostor pairs."
        )
    return genuine_pairs, impostor_pairs


def calculate_similarities(
    embeddings: torch.Tensor, pairs: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    scores = []
    for start in range(0, len(pairs), SIMILARITY_BATCH_SIZE):
        pair_batch = torch.tensor(
            pairs[start : start + SIMILARITY_BATCH_SIZE], dtype=torch.long
        )
        scores.append(
            F.cosine_similarity(
                embeddings[pair_batch[:, 0]],
                embeddings[pair_batch[:, 1]],
                dim=1,
            )
        )
    return torch.cat(scores)


def print_similarity_summary(group_name: str, scores: torch.Tensor) -> None:
    print(f"Number of {group_name} pairs: {scores.numel():,}")
    print(f"Average {group_name} similarity: {scores.mean().item():.6f}")
    print(
        f"Minimum {group_name} similarity: {scores.min().item():.6f}"
    )
    print(
        f"Maximum {group_name} similarity: {scores.max().item():.6f}"
    )


def run_evaluation() -> None:
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    validate_checkpoint_architecture(checkpoint)
    identity_to_label = validate_identity_mapping(checkpoint.get("identity_to_label"))
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint is missing a valid model_state_dict.")

    model = ResNet50(num_classes=len(identity_to_label), pretrained=False)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint weights do not match a {len(identity_to_label)}-class ResNet-50: "
            f"{exc}"
        ) from exc
    model.eval()

    dataset = QMULSurvFaceDataset()
    image_indices, embedding_indices_by_identity, insufficient_count = (
        select_training_images(dataset, identity_to_label)
    )
    loader = DataLoader(
        Subset(dataset, image_indices),
        batch_size=EMBEDDING_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    embedding_batches = []
    with torch.no_grad():
        for images, _ in loader:
            embeddings, _ = model(images)
            embedding_batches.append(embeddings)
    all_embeddings = torch.cat(embedding_batches)
    if all_embeddings.shape != (len(image_indices), ResNet50.embedding_dim):
        raise RuntimeError(
            f"Expected {len(image_indices)} embeddings of dimension "
            f"{ResNet50.embedding_dim}, got {tuple(all_embeddings.shape)}."
        )

    genuine_pairs, impostor_pairs = create_pairs(embedding_indices_by_identity)
    genuine_scores = calculate_similarities(all_embeddings, genuine_pairs)
    impostor_scores = calculate_similarities(all_embeddings, impostor_pairs)

    print(f"Random seed: {RANDOM_SEED}")
    print(
        f"Sampled {len(embedding_indices_by_identity):,} identities and "
        f"{len(image_indices):,} training images."
    )
    print(
        f"Skipped {insufficient_count:,} checkpoint identities with fewer than "
        "two training images."
    )
    print_similarity_summary("genuine", genuine_scores)
    print_similarity_summary("impostor", impostor_scores)
    print("Exploratory training-set results only; no matching threshold is applied.")


def main() -> int:
    try:
        run_evaluation()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Matching evaluation failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
