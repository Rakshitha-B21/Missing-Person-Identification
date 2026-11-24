"""Paginated contact sheets for reviewing saved face images."""

from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np


_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_COLUMNS = 6
_ROWS = 8
_CELL_WIDTH = 180
_IMAGE_HEIGHT = 140
_LABEL_HEIGHT = 30
_HEADER_HEIGHT = 44


def _image_label(path: Path) -> str:
    match = re.search(r"_frame_(\d+)_face_(\d+)$", path.stem)
    if match:
        return f"Frame {match.group(1)} | face_{match.group(2)}"

    match = re.search(r"_face_(\d+)$", path.stem)
    return f"face_{match.group(1)}" if match else path.stem


def _draw_image_tile(sheet: np.ndarray, image_path: Path, x: int, y: int) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read contact sheet image: {image_path}")

    height, width = image.shape[:2]
    scale = min(_CELL_WIDTH / width, _IMAGE_HEIGHT / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    image_x = x + (_CELL_WIDTH - resized_width) // 2
    image_y = y + (_IMAGE_HEIGHT - resized_height) // 2
    sheet[
        image_y : image_y + resized_height,
        image_x : image_x + resized_width,
    ] = resized

    label = _image_label(image_path)
    if len(label) > 30:
        label = f"{label[:27]}..."
    cv2.putText(
        sheet,
        label,
        (x + 4, y + _IMAGE_HEIGHT + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )


def create_contact_sheets(
    images_dir: Path,
    output_dir: Path,
    sheet_prefix: str,
) -> list[Path]:
    """Create labeled, paginated JPEG sheets for images in a directory."""
    if not sheet_prefix or Path(sheet_prefix).name != sheet_prefix:
        raise ValueError("Contact sheet prefix must be a non-empty filename stem.")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {images_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    page_capacity = _COLUMNS * _ROWS
    page_count = max(1, math.ceil(len(images) / page_capacity))
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: list[Path] = []

    for page_index in range(page_count):
        sheet = np.full(
            (
                _HEADER_HEIGHT + _ROWS * (_IMAGE_HEIGHT + _LABEL_HEIGHT),
                _COLUMNS * _CELL_WIDTH,
                3,
            ),
            255,
            dtype=np.uint8,
        )
        title = f"{sheet_prefix} ({page_index + 1}/{page_count})"
        cv2.putText(
            sheet,
            title,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )

        page_images = images[
            page_index * page_capacity : (page_index + 1) * page_capacity
        ]
        if not page_images:
            cv2.putText(
                sheet,
                "No faces found",
                (24, _HEADER_HEIGHT + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (25, 25, 25),
                2,
                cv2.LINE_AA,
            )
        for item_index, image_path in enumerate(page_images):
            column = item_index % _COLUMNS
            row = item_index // _COLUMNS
            x = column * _CELL_WIDTH
            y = _HEADER_HEIGHT + row * (_IMAGE_HEIGHT + _LABEL_HEIGHT)
            _draw_image_tile(sheet, image_path, x, y)

        output_path = output_dir / f"{sheet_prefix}_{page_index + 1:03d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"Could not write contact sheet: {output_path}")
        sheet_paths.append(output_path)

    return sheet_paths
