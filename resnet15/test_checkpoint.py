"""Load a saved ResNet-50 checkpoint and run one training image through it."""

import pickle
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

if __package__:
    from .dataset import QMULSurvFaceDataset
    from .model import ResNet50
else:
    from dataset import QMULSurvFaceDataset
    from model import ResNet50


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "checkpoints" / "resnet50_best_model.pth"
)


def load_checkpoint(checkpoint_path: Path) -> Mapping:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Unable to load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint '{checkpoint_path}' must contain a mapping.")
    return checkpoint


def validate_identity_mapping(mapping: object) -> Mapping:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("Checkpoint is missing a non-empty identity_to_label mapping.")
    if any(not isinstance(identity, str) for identity in mapping):
        raise ValueError("Checkpoint identity_to_label keys must be identity folder names.")

    labels = list(mapping.values())
    if any(not isinstance(label, int) or isinstance(label, bool) for label in labels):
        raise ValueError("Checkpoint identity_to_label values must be integer labels.")
    if set(labels) != set(range(len(labels))):
        raise ValueError(
            "Checkpoint identity_to_label values must be consecutive labels starting at 0."
        )
    return mapping


def validate_checkpoint_architecture(checkpoint: Mapping) -> None:
    architecture = checkpoint.get("architecture")
    if architecture != ResNet50.architecture:
        raise ValueError(
            f"Expected a {ResNet50.architecture} checkpoint, found "
            f"{architecture!r}. ResNet-15 checkpoints cannot be loaded by this model."
        )


def find_checkpoint_image(dataset: QMULSurvFaceDataset, mapping: Mapping) -> int:
    for identity in mapping:
        original_label = dataset.class_to_idx.get(identity)
        if original_label is None:
            continue
        for index, (_, label) in enumerate(dataset.samples):
            if label == original_label:
                return index

    raise ValueError(
        f"No image under '{dataset.root}' matches an identity in the checkpoint."
    )


def run_checkpoint_test() -> None:
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
    image_index = find_checkpoint_image(dataset, identity_to_label)
    image, _ = dataset[image_index]
    input_image = image.unsqueeze(0)

    with torch.no_grad():
        embedding, classifier_output = model(input_image)

    if classifier_output is None:
        raise RuntimeError("The loaded model has no identity classifier.")

    print(f"Input image shape: {tuple(input_image.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Classifier output shape: {tuple(classifier_output.shape)}")


def main() -> int:
    try:
        run_checkpoint_test()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Checkpoint test failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
