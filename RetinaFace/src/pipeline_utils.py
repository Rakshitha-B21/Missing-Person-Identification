"""Shared output-folder, image-saving, and annotation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.detector import FaceDetection


@dataclass(frozen=True)
class OutputDirectories:
    root: Path
    cropped_faces: Path
    high_resolution_faces: Path
    aligned_faces: Path
    annotated_images: Path
    reports: Path


def prepare_output_directories(output_root: Path) -> OutputDirectories:
    directories = OutputDirectories(
        root=output_root,
        cropped_faces=output_root / "cropped_faces",
        high_resolution_faces=output_root / "high_resolution_faces",
        aligned_faces=output_root / "aligned_faces",
        annotated_images=output_root / "annotated_images",
        reports=output_root / "reports",
    )
    for directory in (
        directories.cropped_faces,
        directories.high_resolution_faces,
        directories.aligned_faces,
        directories.annotated_images,
        directories.reports,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_jpeg(path: Path, image: np.ndarray, quality: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    ):
        raise OSError(f"Could not write image: {path}")


def save_png(path: Path, image: np.ndarray, compression: int = 3) -> None:
    if not 0 <= compression <= 9:
        raise ValueError("PNG compression must be between 0 and 9.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, compression],
    ):
        raise OSError(f"Could not write image: {path}")


def annotate_detections(
    image: np.ndarray,
    detections: Iterable[tuple[str, FaceDetection]],
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for face_id, detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        top_left = (max(0, min(width - 1, x1)), max(0, min(height - 1, y1)))
        bottom_right = (
            max(0, min(width - 1, x2 - 1)),
            max(0, min(height - 1, y2 - 1)),
        )
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
        label = f"{face_id} {detection.confidence:.2f}"
        label_y = max(20, top_left[1] - 8)
        cv2.putText(
            annotated,
            label,
            (top_left[0], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated
