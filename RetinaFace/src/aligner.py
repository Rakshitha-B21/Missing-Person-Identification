"""Five-landmark similarity alignment for recognition-ready face images."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import cv2
import numpy as np


_REFERENCE_POINTS_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray,
    landmarks: Mapping[str, Sequence[float]] | None,
    output_size: int = 112,
) -> np.ndarray | None:
    """Align a face to a square template, or return None for invalid landmarks."""
    if output_size <= 0:
        raise ValueError("Alignment output size must be a positive integer.")
    if image is None or image.ndim < 2 or not landmarks:
        return None

    required = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
    try:
        points = {
            name: (float(landmarks[name][0]), float(landmarks[name][1]))
            for name in required
        }
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for point in points.values() for value in point):
        return None

    image_height, image_width = image.shape[:2]
    if image_width <= 0 or image_height <= 0 or any(
        x < 0 or y < 0 or x >= image_width or y >= image_height
        for x, y in points.values()
    ):
        return None

    # RetinaFace's eye/mouth names can refer to anatomical sides. Sort by image x
    # so the first template point always means the viewer's left-hand feature.
    eyes = sorted((points["left_eye"], points["right_eye"]), key=lambda point: point[0])
    if math.dist(*eyes) < 1.0:
        return None
    mouth = sorted(
        (points["mouth_left"], points["mouth_right"]), key=lambda point: point[0]
    )
    source_points = np.asarray(
        [*eyes, points["nose"], *mouth],
        dtype=np.float32,
    )
    destination_points = _REFERENCE_POINTS_112 * (output_size / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(
        source_points,
        destination_points,
        method=cv2.LMEDS,
    )
    if (
        transform is None
        or transform.shape != (2, 3)
        or not np.isfinite(transform).all()
    ):
        return None

    return cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
