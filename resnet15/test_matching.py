"""Compare embeddings from same-identity and different-identity training images."""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn.functional as F

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


def select_image_indices(
    dataset: QMULSurvFaceDataset, identity_to_label: Mapping
) -> Tuple[int, int, int]:
    indices_by_label: Dict[int, List[int]] = defaultdict(list)
    for index, (_, label) in enumerate(dataset.samples):
        indices_by_label[label].append(index)

    selected_identities = [
        (identity, dataset.class_to_idx[identity])
        for identity in sorted(identity_to_label)
        if identity in dataset.class_to_idx
    ]

    same_identity_indices = None
    same_identity_label = None
    for _, label in selected_identities:
        indices = indices_by_label[label]
        if len(indices) < 2:
            continue
        first_index = indices[0]
        second_index = next(
            (
                index
                for index in indices[1:]
                if dataset.samples[index][0] != dataset.samples[first_index][0]
            ),
            None,
        )
        if second_index is not None:
            same_identity_indices = (first_index, second_index)
            same_identity_label = label
            break

    if same_identity_indices is None:
        raise ValueError(
            f"No checkpoint identity has two different images under '{dataset.root}'."
        )

    for _, label in selected_identities:
        if label != same_identity_label and indices_by_label[label]:
            return (
                same_identity_indices[0],
                same_identity_indices[1],
                indices_by_label[label][0],
            )

    raise ValueError(
        f"No image from a second checkpoint identity was found under '{dataset.root}'."
    )


def run_matching_test() -> None:
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
    image_indices = select_image_indices(dataset, identity_to_label)
    images = torch.stack([dataset[index][0] for index in image_indices])

    with torch.no_grad():
        embeddings, classifier_output = model(images)

    if embeddings.shape != (3, ResNet50.embedding_dim):
        raise RuntimeError(
            f"Expected three {ResNet50.embedding_dim}-dimensional embeddings, "
            f"got {tuple(embeddings.shape)}."
        )
    if classifier_output is None:
        raise RuntimeError("The loaded model has no identity classifier.")

    same_identity_similarity = F.cosine_similarity(
        embeddings[0], embeddings[1], dim=0
    ).item()
    different_identity_similarity = F.cosine_similarity(
        embeddings[0], embeddings[2], dim=0
    ).item()

    print(
        f"Same-identity cosine similarity: {same_identity_similarity:.6f}"
    )
    print(
        f"Different-identity cosine similarity: "
        f"{different_identity_similarity:.6f}"
    )


def main() -> int:
    try:
        run_matching_test()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Matching test failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
