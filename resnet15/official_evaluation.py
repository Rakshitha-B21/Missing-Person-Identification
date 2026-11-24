"""Extract ResNet-50 features for a shared-identity SurvFace pilot."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import ResNet50
from test_checkpoint import (
    load_checkpoint,
    validate_checkpoint_architecture,
    validate_identity_mapping,
)


PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "dataset" / "QMUL-SurvFace" / "Face_Identification_Test_Set"
CHECKPOINT = (
    Path(__file__).resolve().parent / "checkpoints" / "resnet50_best_model.pth"
)
OUTPUT = (
    Path(__file__).resolve().parent
    / "official_results"
    / "resnet50_pilot_matched_100"
)
MAX_IDENTITIES = 100
DEFAULT_UNMATED_LIMIT = 1000

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


class FaceDataset(Dataset):
    def __init__(self, folder: Path, filenames: list[str]) -> None:
        self.folder = folder
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.folder / self.filenames[index]
        try:
            with Image.open(path) as image:
                return TRANSFORM(image.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load evaluation image '{path}': {exc}") from exc


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return parsed


def get_mat_data(
    mat_file: Path, filename_key: str, id_key: str
) -> tuple[list[str], np.ndarray]:
    if not mat_file.is_file():
        raise FileNotFoundError(f"Evaluation metadata not found: {mat_file}")
    data = loadmat(mat_file)
    if filename_key not in data or id_key not in data:
        raise ValueError(
            f"Metadata file '{mat_file}' is missing '{filename_key}' or '{id_key}'."
        )

    filenames = [str(item[0]) for item in data[filename_key].ravel()]
    identity_ids = data[id_key].ravel().astype(np.int32)
    if len(filenames) != len(identity_ids):
        raise ValueError(f"Filename and identity counts differ in '{mat_file}'.")
    return filenames, identity_ids


def verify_images_exist(folder: Path, filenames: list[str], split_name: str) -> None:
    missing = [name for name in filenames if not (folder / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} {split_name} images listed by the official metadata "
            f"were not found under '{folder}'. Examples: {preview}"
        )


def select_pilot_data(
    max_unmated: int, seed: int
) -> tuple[
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    gallery_names, gallery_ids = get_mat_data(
        DATASET / "gallery_img_ID_pairs.mat", "gallery_set", "gallery_ids"
    )
    mated_names, mated_ids = get_mat_data(
        DATASET / "mated_probe_img_ID_pairs.mat",
        "mated_probe_set",
        "mated_probe_ids",
    )

    shared_ids = np.intersect1d(np.unique(gallery_ids), np.unique(mated_ids))
    if shared_ids.size == 0:
        raise ValueError("Gallery and mated probes have no shared identities.")
    identity_rng = np.random.default_rng(seed)
    selected_ids = np.sort(
        identity_rng.choice(
            shared_ids, size=min(MAX_IDENTITIES, shared_ids.size), replace=False
        )
    ).astype(np.int32)

    selected_gallery_mask = np.isin(gallery_ids, selected_ids)
    selected_mated_mask = np.isin(mated_ids, selected_ids)
    selected_gallery_names = [
        name for name, selected in zip(gallery_names, selected_gallery_mask) if selected
    ]
    selected_gallery_ids = gallery_ids[selected_gallery_mask]
    selected_mated_names = [
        name for name, selected in zip(mated_names, selected_mated_mask) if selected
    ]
    selected_mated_ids = mated_ids[selected_mated_mask]

    unmated_dir = DATASET / "unmated_probe"
    if not unmated_dir.is_dir():
        raise NotADirectoryError(f"Unmated probe directory not found: {unmated_dir}")
    available_unmated = sorted(
        path.name for path in unmated_dir.glob("*.jpg") if path.is_file()
    )
    unmated_count = min(max_unmated, len(available_unmated))
    if unmated_count:
        unmated_rng = np.random.default_rng(seed + 1)
        selected_positions = np.sort(
            unmated_rng.choice(
                len(available_unmated), size=unmated_count, replace=False
            )
        )
        selected_unmated_names = [
            available_unmated[int(position)] for position in selected_positions
        ]
    else:
        selected_unmated_names = []
    selected_unmated_ids = np.full(
        len(selected_unmated_names), -100, dtype=np.int32
    )

    return (
        selected_gallery_names,
        selected_gallery_ids,
        selected_mated_names,
        selected_mated_ids,
        selected_unmated_names,
        selected_unmated_ids,
        selected_ids,
    )


def save_filenames(output_folder: Path, name: str, filenames: list[str]) -> None:
    with (output_folder / f"{name}_filenames.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        if filenames:
            file.write("\n".join(filenames) + "\n")


def extract_features(
    model: ResNet50,
    folder: Path,
    filenames: list[str],
    batch_size: int,
    output_folder: Path,
    split_name: str,
) -> None:
    dataset = FaceDataset(folder, filenames)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    feature_path = output_folder / f"{split_name}_features.npy"
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), ResNet50.embedding_dim),
    )
    position = 0
    with torch.inference_mode():
        for batch in loader:
            embeddings, _ = model(batch)
            embeddings = embeddings.cpu().numpy()
            end = position + len(embeddings)
            features[position:end] = embeddings
            position = end
            print(
                f"\r{split_name}: {position}/{len(dataset)} images",
                end="",
                flush=True,
            )
    if position != len(dataset):
        raise RuntimeError(
            f"Extracted {position} {split_name} features for {len(dataset)} images."
        )
    features.flush()
    del features
    print(f"\nSaved: {feature_path}")


def run_extraction(batch_size: int, max_unmated: int, seed: int) -> None:
    if not DATASET.is_dir():
        raise NotADirectoryError(f"Official test-set directory not found: {DATASET}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"ResNet-50 checkpoint not found: {CHECKPOINT}")
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output folder already exists: {OUTPUT}\n"
            "Existing evaluation results are never overwritten or deleted."
        )

    (
        gallery_names,
        gallery_ids,
        mated_names,
        mated_ids,
        unmated_names,
        unmated_ids,
        selected_ids,
    ) = select_pilot_data(max_unmated, seed)

    if not len(gallery_names) or not len(mated_names):
        raise ValueError("Selected identities produced an empty gallery or mated set.")
    verify_images_exist(DATASET / "gallery", gallery_names, "gallery")
    verify_images_exist(DATASET / "mated_probe", mated_names, "mated probe")
    verify_images_exist(DATASET / "unmated_probe", unmated_names, "unmated probe")

    overlap = np.intersect1d(np.unique(gallery_ids), np.unique(mated_ids))
    if not np.array_equal(overlap, selected_ids):
        raise RuntimeError(
            "Selected gallery and mated-probe identities do not match exactly."
        )

    checkpoint = load_checkpoint(CHECKPOINT)
    validate_checkpoint_architecture(checkpoint)
    identity_mapping = validate_identity_mapping(checkpoint.get("identity_to_label"))
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint is missing a valid model_state_dict.")
    model = ResNet50(
        num_classes=len(identity_mapping), pretrained=False
    ).to(torch.device("cpu"))
    try:
        model.load_state_dict(model_state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint weights do not match a {len(identity_mapping)}-class "
            f"ResNet-50 model: {exc}"
        ) from exc
    model.eval()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".resnet50_pilot_matched_100-", dir=OUTPUT.parent)
    )
    published = False
    try:
        print(f"Selected identities: {len(selected_ids)}")
        print(f"Gallery images: {len(gallery_names)}")
        print(f"Mated probes: {len(mated_names)}")
        print(f"Unmated probes: {len(unmated_names)}")
        print("Device: cpu")
        print(f"Batch size: {batch_size}")

        splits = (
            ("gallery", gallery_names, gallery_ids),
            ("mated_probe", mated_names, mated_ids),
            ("unmated_probe", unmated_names, unmated_ids),
        )
        for name, filenames, ids in splits:
            np.save(staging_dir / f"{name}_ids.npy", ids.astype(np.int32))
            save_filenames(staging_dir, name, filenames)
            extract_features(
                model,
                DATASET / name,
                filenames,
                batch_size,
                staging_dir,
                name,
            )

        np.save(staging_dir / "selected_identity_ids.npy", selected_ids)
        if OUTPUT.exists():
            raise FileExistsError(
                f"Output folder appeared during extraction; preserving it: {OUTPUT}"
            )
        staging_dir.rename(OUTPUT)
        published = True
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nPILOT FEATURE EXTRACTION COMPLETED.")
    print("Results folder:", OUTPUT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract features for up to 100 identities shared by the official "
            "gallery and mated-probe metadata."
        )
    )
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--max-unmated", type=positive_int, default=DEFAULT_UNMATED_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        run_extraction(args.batch_size, args.max_unmated, args.seed)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Pilot feature extraction failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
