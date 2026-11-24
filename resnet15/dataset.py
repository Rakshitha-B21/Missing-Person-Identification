"""PyTorch dataset for the QMUL-SurvFace training split."""

from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_TRAINING_ROOT = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "QMUL-SurvFace"
    / "training_set"
)


class QMULSurvFaceDataset(Dataset):
    """Load training images and map each identity folder to a stable label.

    Only images beneath ``root`` are indexed; the default root points directly
    to ``dataset/QMUL-SurvFace/training_set``.
    """

    def __init__(self, root: Path = DEFAULT_TRAINING_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Training-set directory not found: {self.root}")

        identity_dirs = sorted(
            (path for path in self.root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        self.class_to_idx = {
            identity_dir.name: index
            for index, identity_dir in enumerate(identity_dirs)
        }

        self.samples = []
        for identity_dir in identity_dirs:
            label = self.class_to_idx[identity_dir.name]
            image_paths = sorted(
                path
                for path in identity_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".jpg"
            )
            self.samples.extend((path, label) for path in image_paths)

        if not self.samples:
            raise ValueError(f"No .jpg training images found under: {self.root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Unable to read image '{image_path}': {exc}") from exc

        return image_tensor, label


if __name__ == "__main__":
    dataset = QMULSurvFaceDataset()
    image_tensor, identity_label = dataset[0]
    print(f"image shape: {tuple(image_tensor.shape)}, label: {identity_label}")