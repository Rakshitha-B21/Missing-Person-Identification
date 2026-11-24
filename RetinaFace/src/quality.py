"""Original-resolution face quality measurements and review flags."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


SMALL_FACE_DIMENSION_PIXELS = 32
BLUR_VARIANCE_THRESHOLD = 1000.0
UNDEREXPOSED_BRIGHTNESS_THRESHOLD = 50.0
OVEREXPOSED_BRIGHTNESS_THRESHOLD = 205.0
LOW_CONTRAST_THRESHOLD = 15.0


@dataclass(frozen=True)
class FaceQualityMetrics:
    blur_score: float | None
    brightness_mean: float | None
    contrast_stddev: float | None
    flags: tuple[str, ...]


def measure_face_quality(
    image: np.ndarray,
    bounding_box: tuple[int, int, int, int],
) -> FaceQualityMetrics:
    """Measure focus and illumination within the original, clipped face box."""
    if image is None or image.ndim < 2:
        raise ValueError("A valid image is required to measure face quality.")
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bounding_box
    face_width = x2 - x1
    face_height = y2 - y1
    flags: list[str] = []

    if face_width < SMALL_FACE_DIMENSION_PIXELS or face_height < SMALL_FACE_DIMENSION_PIXELS:
        flags.append("small_face")
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        flags.append("partially_cropped")

    clipped_x1 = max(0, min(image_width, x1))
    clipped_y1 = max(0, min(image_height, y1))
    clipped_x2 = max(0, min(image_width, x2))
    clipped_y2 = max(0, min(image_height, y2))
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return FaceQualityMetrics(None, None, None, tuple((*flags, "invalid_face_region")))

    face = image[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    if face.ndim == 2:
        gray = face
    elif face.shape[2] == 3:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    elif face.shape[2] == 1:
        gray = face[:, :, 0]
    else:
        raise ValueError("Face quality measurements require a grayscale or BGR image.")
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = (
        float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if gray.shape[0] >= 3 and gray.shape[1] >= 3
        else 0.0
    )

    if blur < BLUR_VARIANCE_THRESHOLD:
        flags.append("blurry")
    if brightness < UNDEREXPOSED_BRIGHTNESS_THRESHOLD:
        flags.append("underexposed")
    elif brightness > OVEREXPOSED_BRIGHTNESS_THRESHOLD:
        flags.append("overexposed")
    if contrast < LOW_CONTRAST_THRESHOLD:
        flags.append("low_contrast")

    return FaceQualityMetrics(
        blur_score=round(blur, 2) if math.isfinite(blur) else None,
        brightness_mean=round(brightness, 2) if math.isfinite(brightness) else None,
        contrast_stddev=round(contrast, 2) if math.isfinite(contrast) else None,
        flags=tuple(flags),
    )
