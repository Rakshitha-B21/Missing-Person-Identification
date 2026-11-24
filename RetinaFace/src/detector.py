"""RetinaFace inference and normalization of its detection results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from retinaface import RetinaFace


@dataclass(frozen=True)
class FaceDetection:
    """A face box, its detector confidence, and optional five-point landmarks."""

    bounding_box: tuple[int, int, int, int]
    confidence: float
    landmarks: dict[str, tuple[float, float]] | None


_LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth_left",
    "mouth_right",
)


def validate_threshold(threshold: float) -> None:
    """Raise when a confidence threshold is outside the supported range."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Detection threshold must be between 0.0 and 1.0.")


def _parse_landmarks(value: Any) -> dict[str, tuple[float, float]] | None:
    if not isinstance(value, Mapping):
        return None

    parsed: dict[str, tuple[float, float]] = {}
    try:
        for name in _LANDMARK_NAMES:
            point = value[name]
            if len(point) != 2:
                return None
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            parsed[name] = (x, y)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return parsed


def _parse_detection(value: Any) -> FaceDetection | None:
    if not isinstance(value, Mapping):
        return None

    try:
        area = value["facial_area"]
        if len(area) != 4:
            return None
        x1, y1, x2, y2 = (int(coordinate) for coordinate in area)
        confidence = float(value["score"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    if x2 <= x1 or y2 <= y1 or not math.isfinite(confidence):
        return None

    return FaceDetection(
        bounding_box=(x1, y1, x2, y2),
        confidence=confidence,
        landmarks=_parse_landmarks(value.get("landmarks")),
    )


def detect_faces_with_raw_count(
    image: Any,
    threshold: float = 0.80,
) -> tuple[list[FaceDetection], int]:
    """Return parsed detections and the number of raw RetinaFace result entries."""
    validate_threshold(threshold)
    result = RetinaFace.detect_faces(image, threshold=threshold)
    if not isinstance(result, dict):
        return [], 0

    detections: list[FaceDetection] = []
    for raw_detection in result.values():
        detection = _parse_detection(raw_detection)
        if detection is not None and detection.confidence >= threshold:
            detections.append(detection)
    return detections, len(result)


def detect_faces(image: Any, threshold: float = 0.80) -> list[FaceDetection]:
    """Detect faces in a BGR image and return valid results above threshold."""
    return detect_faces_with_raw_count(image, threshold)[0]
