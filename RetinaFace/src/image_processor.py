"""Batch processing of supported image files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.aligner import align_face
from src.cropper import crop_face, validate_margins
from src.detector import detect_faces, validate_threshold
from src.pipeline_utils import (
    annotate_detections,
    prepare_output_directories,
    save_jpeg,
    save_png,
)
from src.quality import measure_face_quality
from src.reporter import DetectionRecord


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageProcessingSummary:
    images_processed: int = 0
    faces_detected: int = 0
    images_with_no_faces: list[str] = field(default_factory=list)
    records: list[DetectionRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _output_prefix(image_path: Path, image_root: Path) -> str:
    return "_".join(image_path.relative_to(image_root).parts)


def process_images(
    image_dir: Path,
    output_dir: Path,
    threshold: float = 0.80,
    margin_x: float = 0.30,
    margin_y: float = 0.30,
    alignment_size: int = 112,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
) -> ImageProcessingSummary:
    """Detect, crop, align, annotate, and report on images under image_dir."""
    validate_threshold(threshold)
    validate_margins(margin_x, margin_y)
    if margin_top is not None:
        validate_margins(0.0, margin_top)
    if margin_bottom is not None:
        validate_margins(0.0, margin_bottom)
    if alignment_size <= 0:
        raise ValueError("Alignment output size must be a positive integer.")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")

    output = prepare_output_directories(output_dir)
    summary = ImageProcessingSummary()
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    existing_outputs = [
        existing_output
        for image_path in image_paths
        for folder in (
            output.cropped_faces,
            output.high_resolution_faces,
            output.aligned_faces,
            output.annotated_images,
        )
        for existing_output in folder.iterdir()
        if existing_output.is_file()
        and existing_output.name.startswith(f"{_output_prefix(image_path, image_dir)}_")
    ]
    if existing_outputs:
        raise FileExistsError(
            "Image outputs already exist; use a separate output directory to "
            f"preserve them: {existing_outputs[0]}"
        )

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        relative_name = image_path.relative_to(image_dir).as_posix()
        if image is None:
            message = f"Could not read image: {image_path}"
            summary.errors.append(message)
            print(f"Warning: {message}")
            continue

        summary.images_processed += 1
        detections = detect_faces(image, threshold)
        summary.faces_detected += len(detections)
        if not detections:
            summary.images_with_no_faces.append(relative_name)

        face_entries = [
            (f"face_{index:03d}", detection)
            for index, detection in enumerate(detections, start=1)
        ]
        annotated = annotate_detections(image, face_entries)
        prefix = _output_prefix(image_path, image_dir)

        for face_id, detection in face_entries:
            crop_result = crop_face(
                image,
                detection.bounding_box,
                margin_x=margin_x,
                margin_y=margin_y,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
            )
            if crop_result is None:
                message = f"Invalid crop for {relative_name} ({face_id})"
                summary.errors.append(message)
                print(f"Warning: {message}")
                continue
            crop, _ = crop_result
            crop_name = f"{prefix}_{face_id}.jpg"
            save_jpeg(output.cropped_faces / crop_name, crop)
            high_resolution_crop_name = f"{prefix}_{face_id}.png"
            save_png(
                output.high_resolution_faces / high_resolution_crop_name,
                crop,
            )

            aligned_name = ""
            aligned = align_face(image, detection.landmarks, alignment_size)
            if aligned is not None:
                aligned_name = f"{prefix}_{face_id}.jpg"
                save_jpeg(output.aligned_faces / aligned_name, aligned)

            quality = measure_face_quality(image, detection.bounding_box)
            quality_flags = list(quality.flags)
            if detection.landmarks is None:
                quality_flags.append("missing_landmarks")
            elif aligned is None:
                quality_flags.append("alignment_unavailable")

            summary.records.append(
                DetectionRecord(
                    media_name=relative_name,
                    frame_number=None,
                    timestamp="",
                    face_id=face_id,
                    confidence_score=detection.confidence,
                    bounding_box=detection.bounding_box,
                    crop_filename=f"cropped_faces/{crop_name}",
                    aligned_filename=(
                        f"aligned_faces/{aligned_name}" if aligned_name else ""
                    ),
                    high_resolution_crop_filename=(
                        f"high_resolution_faces/{high_resolution_crop_name}"
                    ),
                    blur_score=quality.blur_score,
                    brightness_mean=quality.brightness_mean,
                    contrast_stddev=quality.contrast_stddev,
                    quality_flags=tuple(quality_flags),
                )
            )

        annotation_name = f"{prefix}_annotated.jpg"
        save_jpeg(output.annotated_images / annotation_name, annotated, quality=95)

    return summary
