"""Reusable sampled-frame video processing for future CCTV footage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.aligner import align_face
from src.cropper import crop_face, validate_margins
from src.detector import (
    FaceDetection,
    detect_faces_with_raw_count,
    validate_threshold,
)
from src.pipeline_utils import (
    annotate_detections,
    prepare_output_directories,
    save_jpeg,
    save_png,
)
from src.quality import measure_face_quality
from src.reporter import DetectionRecord


@dataclass
class VideoProcessingResult:
    frames_sampled: int = 0
    raw_detections: int = 0
    filtered_detections: int = 0
    saved_crops: int = 0
    records: list[DetectionRecord] = field(default_factory=list)

    @property
    def faces_detected(self) -> int:
        """Compatibility alias for detections retained after filtering."""
        return self.filtered_detections


def filter_video_detections(
    detections: list[FaceDetection],
    frame_width: int,
    frame_height: int,
    min_face_width: int = 0,
    min_face_height: int = 0,
) -> list[FaceDetection]:
    """Keep positive boxes that intersect the frame and meet optional size limits."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if min_face_width < 0 or min_face_height < 0:
        raise ValueError("Minimum face dimensions must be non-negative.")

    kept: list[FaceDetection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        width, height = x2 - x1, y2 - y1
        if (
            width <= 0
            or height <= 0
            or x2 <= 0
            or y2 <= 0
            or x1 >= frame_width
            or y1 >= frame_height
            or width < min_face_width
            or height < min_face_height
        ):
            continue
        kept.append(detection)
    return kept

def _format_timestamp(milliseconds: float) -> str:
    total_ms = max(0, round(milliseconds))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def process_video(
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.80,
    margin_x: float = 0.30,
    margin_y: float = 0.30,
    alignment_size: int = 112,
    frame_interval: int = 30,
    min_face_width: int = 0,
    min_face_height: int = 0,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
) -> VideoProcessingResult:
    """Process every Nth video frame and return records for the shared report."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if frame_interval < 1:
        raise ValueError("Frame sampling interval must be at least 1.")
    validate_threshold(threshold)
    validate_margins(margin_x, margin_y)
    if margin_top is not None:
        validate_margins(0.0, margin_top)
    if margin_bottom is not None:
        validate_margins(0.0, margin_bottom)
    if alignment_size <= 0:
        raise ValueError("Alignment output size must be a positive integer.")
    if min_face_width < 0 or min_face_height < 0:
        raise ValueError("Minimum face dimensions must be non-negative.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video_path}")

    output = prepare_output_directories(output_dir)
    previous_frame_prefix = f"{video_path.name}_frame_"
    output_folders = (
        output.cropped_faces,
        output.high_resolution_faces,
        output.aligned_faces,
        output.annotated_images,
    )
    existing_outputs = [
        existing_output
        for folder in output_folders
        for existing_output in folder.iterdir()
        if (
            existing_output.is_file()
            and existing_output.name.startswith(previous_frame_prefix)
            and existing_output.suffix.lower() in {".jpg", ".png"}
        )
    ]
    if existing_outputs:
        capture.release()
        raise FileExistsError(
            f"Video outputs already exist for {video_path.name}; use a separate "
            f"output directory to preserve them: {existing_outputs[0]}"
        )

    result = VideoProcessingResult()
    fps = capture.get(cv2.CAP_PROP_FPS)
    has_valid_fps = math.isfinite(fps) and fps > 0
    frames_read = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames_read += 1
            reported_next_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if (
                math.isfinite(reported_next_frame)
                and reported_next_frame >= 1
                and math.isclose(reported_next_frame, round(reported_next_frame))
            ):
                frame_index = round(reported_next_frame) - 1
            else:
                frame_index = frames_read - 1

            if frame_index % frame_interval == 0:
                frame_number = frame_index + 1
                result.frames_sampled += 1
                if has_valid_fps:
                    timestamp = _format_timestamp(frame_index * 1000.0 / fps)
                else:
                    timestamp = ""

                raw_detections, raw_count = detect_faces_with_raw_count(
                    frame,
                    threshold,
                )
                result.raw_detections += raw_count
                detections = filter_video_detections(
                    raw_detections,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    min_face_width=min_face_width,
                    min_face_height=min_face_height,
                )
                result.filtered_detections += len(detections)
                face_entries = [
                    (f"face_{index:03d}", detection)
                    for index, detection in enumerate(detections, start=1)
                ]
                annotated = annotate_detections(frame, face_entries)
                frame_prefix = f"{video_path.name}_frame_{frame_number:06d}"

                for face_id, detection in face_entries:
                    crop_result = crop_face(
                        frame,
                        detection.bounding_box,
                        margin_x=margin_x,
                        margin_y=margin_y,
                        margin_top=margin_top,
                        margin_bottom=margin_bottom,
                    )
                    if crop_result is None:
                        print(
                            f"Warning: Invalid crop in {video_path.name}, "
                            f"frame {frame_number} ({face_id})"
                        )
                        continue
                    crop, _ = crop_result
                    crop_name = f"{frame_prefix}_{face_id}.jpg"
                    save_jpeg(output.cropped_faces / crop_name, crop)
                    result.saved_crops += 1
                    high_resolution_crop_name = f"{frame_prefix}_{face_id}.png"
                    save_png(
                        output.high_resolution_faces / high_resolution_crop_name,
                        crop,
                    )

                    aligned_name = ""
                    aligned = align_face(frame, detection.landmarks, alignment_size)
                    if aligned is not None:
                        aligned_name = f"{frame_prefix}_{face_id}.jpg"
                        save_jpeg(output.aligned_faces / aligned_name, aligned)

                    quality = measure_face_quality(frame, detection.bounding_box)
                    quality_flags = list(quality.flags)
                    if detection.landmarks is None:
                        quality_flags.append("missing_landmarks")
                    elif aligned is None:
                        quality_flags.append("alignment_unavailable")

                    result.records.append(
                        DetectionRecord(
                            media_name=video_path.name,
                            frame_number=frame_number,
                            timestamp=timestamp,
                            face_id=face_id,
                            confidence_score=detection.confidence,
                            bounding_box=detection.bounding_box,
                            crop_filename=f"cropped_faces/{crop_name}",
                            aligned_filename=(
                                f"aligned_faces/{aligned_name}" if aligned_name else ""
                            ),
                            high_resolution_crop_filename=(
                                f"high_resolution_faces/"
                                f"{high_resolution_crop_name}"
                            ),
                            blur_score=quality.blur_score,
                            brightness_mean=quality.brightness_mean,
                            contrast_stddev=quality.contrast_stddev,
                            quality_flags=tuple(quality_flags),
                        )
                    )

                annotation_name = f"{frame_prefix}_annotated.jpg"
                save_jpeg(
                    output.annotated_images / annotation_name,
                    annotated,
                    quality=95,
                )

    finally:
        capture.release()

    if frames_read == 0:
        raise OSError(f"Video contains no readable frames: {video_path}")
    return result
