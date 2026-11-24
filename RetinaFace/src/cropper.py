"""Margin-based face crops that retain the source image's aspect ratio."""

from __future__ import annotations

import math
from typing import Any


def validate_margins(margin_x: float, margin_y: float) -> None:
    if (
        not math.isfinite(margin_x)
        or not math.isfinite(margin_y)
        or margin_x < 0
        or margin_y < 0
    ):
        raise ValueError("Crop margins must be finite, non-negative proportions.")


def crop_face(
    image: Any,
    bounding_box: tuple[int, int, int, int],
    margin_x: float = 0.30,
    margin_y: float = 0.30,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
) -> tuple[Any, tuple[int, int, int, int]] | None:
    """Return a clamped, expanded crop and its exclusive-end crop coordinates."""
    top_margin = margin_y + 0.05 if margin_top is None else margin_top
    bottom_margin = margin_y if margin_bottom is None else margin_bottom
    validate_margins(margin_x, top_margin)
    validate_margins(0.0, bottom_margin)
    if image is None or getattr(image, "ndim", 0) < 2:
        raise ValueError("A valid image is required for cropping.")

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bounding_box
    if x2 <= x1 or y2 <= y1 or image_width <= 0 or image_height <= 0:
        return None

    face_width = x2 - x1
    face_height = y2 - y1
    horizontal_margin = round(face_width * margin_x)
    top_padding = round(face_height * top_margin)
    bottom_padding = round(face_height * bottom_margin)

    crop_x1 = max(0, x1 - horizontal_margin)
    crop_y1 = max(0, y1 - top_padding)
    crop_x2 = min(image_width, x2 + horizontal_margin)
    crop_y2 = min(image_height, y2 + bottom_padding)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        return None
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)
