"""Plot genuine and impostor cosine similarities from sampled training images."""

import sys
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

if __package__:
    from .dataset import QMULSurvFaceDataset
    from .evaluate_matching import (
        EMBEDDING_BATCH_SIZE,
        MAX_IDENTITIES,
        RANDOM_SEED,
        calculate_similarities,
        create_pairs,
        select_training_images,
    )
    from .model import ResNet50
    from .test_checkpoint import (
        load_checkpoint,
        validate_checkpoint_architecture,
        validate_identity_mapping,
    )
else:
    from dataset import QMULSurvFaceDataset
    from evaluate_matching import (
        EMBEDDING_BATCH_SIZE,
        MAX_IDENTITIES,
        RANDOM_SEED,
        calculate_similarities,
        create_pairs,
        select_training_images,
    )
    from model import ResNet50
    from test_checkpoint import (
        load_checkpoint,
        validate_checkpoint_architecture,
        validate_identity_mapping,
    )


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "checkpoints" / "resnet50_best_model.pth"
)
PLOT_PATH = (
    Path(__file__).resolve().parent / "similarity_distribution_resnet50.png"
)


def run_plot() -> None:
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
    selected_identity_count = len(embedding_indices_by_identity)
    if selected_identity_count != MAX_IDENTITIES:
        raise ValueError(
            f"Could sample only {selected_identity_count} identities with at least "
            f"two images; exactly {MAX_IDENTITIES} are required."
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

    figure, axis = plt.subplots()
    axis.hist(genuine_scores.numpy(), bins=50, alpha=0.6, label="Genuine pairs")
    axis.hist(impostor_scores.numpy(), bins=50, alpha=0.6, label="Impostor pairs")
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Pair count")
    axis.set_title("Genuine and Impostor Similarity Distributions")
    axis.legend()
    figure.tight_layout()
    try:
        figure.savefig(PLOT_PATH)
    except OSError as exc:
        raise RuntimeError(f"Could not save similarity plot to '{PLOT_PATH}': {exc}") from exc
    finally:
        plt.close(figure)

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Sampled identities: {selected_identity_count}")
    print(f"Skipped identities with fewer than two images: {insufficient_count}")
    print(f"Genuine pairs used: {len(genuine_pairs):,}")
    print(f"Impostor pairs used: {len(impostor_pairs):,}")
    print(f"Saved similarity plot to {PLOT_PATH}")


def main() -> int:
    try:
        run_plot()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Similarity plot failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
