"""A compact custom ResNet-15 for face embeddings and identity classification.

Layer-count convention (when ``num_classes`` is set): stem convolution (1),
the two main-path convolutions in each of six residual blocks (12), embedding
projection (1), and identity classifier (1) total 15 layers. Shortcut
projection convolutions, normalization layers, pooling, and activations are
not included in this convention.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class ResNet15(nn.Module):
    """Custom ResNet-15 that returns normalized face embeddings and logits.

    Args:
        num_classes: Number of identities for the classification head. If
            ``None``, the classifier is omitted and logits are returned as
            ``None``.

    Input images are RGB tensors with shape ``(batch, 3, height, width)``;
    ``112 x 112`` is the intended input size.
    """

    embedding_dim = 512

    def __init__(self, num_classes: Optional[int] = None) -> None:
        super().__init__()
        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be a positive integer or None")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(256, self.embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(self.embedding_dim)
        self.classifier = (
            nn.Linear(self.embedding_dim, num_classes) if num_classes is not None else None
        )

        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                "inputs must be an RGB tensor with shape (batch, 3, height, width)"
            )

        output = self.stem(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.pool(output).flatten(1)

        embeddings = F.normalize(self.embedding_bn(self.embedding(output)), p=2, dim=1)
        logits = self.classifier(embeddings) if self.classifier is not None else None
        return embeddings, logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet15(num_classes=10).to(device).eval()
    dummy_input = torch.randn(2, 3, 112, 112, device=device)

    with torch.no_grad():
        embeddings, logits = model(dummy_input)

    assert embeddings.shape == (2, 512)
    assert logits is not None and logits.shape == (2, 10)
    print(f"embeddings: {tuple(embeddings.shape)}, logits: {tuple(logits.shape)}")
