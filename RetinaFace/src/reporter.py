"""CSV and plain-text reporting for image and video detections."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.quality import SMALL_FACE_DIMENSION_PIXELS


@dataclass(frozen=True)
class DetectionRecord:
    media_name: str
    frame_number: int | None
    timestamp: str
    face_id: str
    confidence_score: float
    bounding_box: tuple[int, int, int, int]
    crop_filename: str
    aligned_filename: str
    high_resolution_crop_filename: str = ""
    blur_score: float | None = None
    brightness_mean: float | None = None
    contrast_stddev: float | None = None
    quality_flags: tuple[str, ...] = ()

    @property
    def original_face_width(self) -> int:
        x1, _, x2, _ = self.bounding_box
        return x2 - x1

    @property
    def original_face_height(self) -> int:
        _, y1, _, y2 = self.bounding_box
        return y2 - y1

    @property
    def is_small_face(self) -> bool:
        """Flag boxes under 32px in either dimension for review, not filtering."""
        return (
            self.original_face_width < SMALL_FACE_DIMENSION_PIXELS
            or self.original_face_height < SMALL_FACE_DIMENSION_PIXELS
        )


_CSV_FIELDS = (
    "image_video_name",
    "frame_number",
    "timestamp",
    "face_id",
    "confidence_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "original_face_width",
    "original_face_height",
    "crop_filename",
    "high_resolution_crop_filename",
    "aligned_filename",
    "blur_score",
    "brightness_mean",
    "contrast_stddev",
    "small_face",
    "quality_flags",
)


def write_csv_report(records: list[DetectionRecord], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            x1, y1, x2, y2 = record.bounding_box
            writer.writerow(
                {
                    "image_video_name": record.media_name,
                    "frame_number": (
                        "" if record.frame_number is None else record.frame_number
                    ),
                    "timestamp": record.timestamp,
                    "face_id": record.face_id,
                    "confidence_score": f"{record.confidence_score:.6f}",
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "original_face_width": record.original_face_width,
                    "original_face_height": record.original_face_height,
                    "crop_filename": record.crop_filename,
                    "high_resolution_crop_filename": (
                        record.high_resolution_crop_filename
                    ),
                    "aligned_filename": record.aligned_filename,
                    "blur_score": (
                        "" if record.blur_score is None else record.blur_score
                    ),
                    "brightness_mean": (
                        ""
                        if record.brightness_mean is None
                        else record.brightness_mean
                    ),
                    "contrast_stddev": (
                        ""
                        if record.contrast_stddev is None
                        else record.contrast_stddev
                    ),
                    "small_face": record.is_small_face,
                    "quality_flags": ";".join(record.quality_flags),
                }
            )


def write_summary(
    report_path: Path,
    images_processed: int,
    faces_detected: int,
    images_with_no_faces: list[str],
    video_frames_sampled: int | None = None,
    video_raw_detections: int = 0,
    video_filtered_detections: int = 0,
    video_saved_crops: int = 0,
    small_faces_count: int = 0,
    quality_flag_counts: dict[str, int] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Total images processed: {images_processed}",
        f"Total faces detected: {faces_detected}",
        f"Very small faces (under 32 px in either box dimension): {small_faces_count}",
        f"Images with no detected faces: {len(images_with_no_faces)}",
    ]
    if quality_flag_counts:
        lines.append("Face quality flags:")
        lines.extend(
            f"- {flag}: {count}"
            for flag, count in sorted(quality_flag_counts.items())
        )
    if images_with_no_faces:
        lines.append("No-face images:")
        lines.extend(f"- {name}" for name in images_with_no_faces)
    if video_frames_sampled is not None:
        lines.extend(
            [
                f"Video total sampled frames: {video_frames_sampled}",
                f"Video raw detections: {video_raw_detections}",
                f"Video filtered detections: {video_filtered_detections}",
                f"Video saved crops: {video_saved_crops}",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
