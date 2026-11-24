"""ResNet-50 face embeddings and identity classification for QMUL-SurvFace."""

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    """ImageNet-initialized ResNet-50 with a normalized 512-D face embedding.

    Args:
        num_classes: Number of training identities. If ``None``, omit the
            identity classifier and return ``None`` for logits.
        pretrained: Load the default ImageNet weights when available. If those
            weights cannot be retrieved, initialize the backbone randomly and
            emit a warning.
    """

    embedding_dim = 512
    architecture = "resnet50"

    def __init__(
        self, num_classes: Optional[int] = None, pretrained: bool = True
    ) -> None:
        super().__init__()
        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be a positive integer or None")

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = resnet50(weights=weights)
        except (OSError, RuntimeError) as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load ImageNet ResNet-50 weights ({exc}); "
                "using randomly initialized backbone weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.backbone = resnet50(weights=None)

        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(2048, self.embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(self.embedding_dim)
        self.classifier = (
            nn.Linear(self.embedding_dim, num_classes)
            if num_classes is not None
            else None
        )

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                "inputs must be an RGB tensor with shape (batch, 3, height, width)"
            )

        features = self.backbone(inputs)
        embeddings = F.normalize(
            self.embedding_bn(self.embedding(features)), p=2, dim=1
        )
        logits = self.classifier(embeddings) if self.classifier is not None else None
        return embeddings, logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(num_classes=10, pretrained=False).to(device).eval()
    dummy_input = torch.randn(2, 3, 112, 112, device=device)

    with torch.no_grad():
        embeddings, logits = model(dummy_input)

    assert embeddings.shape == (2, 512)
    assert logits is not None and logits.shape == (2, 10)
    print(f"embeddings: {tuple(embeddings.shape)}, logits: {tuple(logits.shape)}")
