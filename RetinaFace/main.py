"""Run RetinaFace over the images folder and, optionally, a video file."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.image_processor import process_images
from src.reporter import DetectionRecord, write_csv_report, write_summary
from src.video_processor import process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, crop, align, and annotate faces with RetinaFace."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--margin-x", type=float, default=0.30)
    parser.add_argument("--margin-y", type=float, default=0.30)
    parser.add_argument(
        "--margin-top",
        type=float,
        help="Optional top crop padding proportion (default: margin-y + 0.05).",
    )
    parser.add_argument(
        "--margin-bottom",
        type=float,
        help="Optional bottom crop padding proportion (default: margin-y).",
    )
    parser.add_argument("--alignment-size", type=int, default=112)
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional video to process in addition to the images folder.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Detect every Nth video frame (default: 30).",
    )
    parser.add_argument(
        "--min-face-width",
        type=int,
        default=0,
        help="Minimum video face-box width in pixels (default: 0; use 0 to disable).",
    )
    parser.add_argument(
        "--min-face-height",
        type=int,
        default=0,
        help="Minimum video face-box height in pixels (default: 0; use 0 to disable).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_dir = args.output_dir / "reports"
    existing_reports = [
        path
        for path in (report_dir / "detections.csv", report_dir / "summary.txt")
        if path.exists()
    ]
    if existing_reports:
        print(
            "Error: Report outputs already exist; use a separate output directory "
            f"to preserve them: {existing_reports[0]}"
        )
        return 1

    try:
        image_summary = process_images(
            args.images_dir,
            args.output_dir,
            threshold=args.threshold,
            margin_x=args.margin_x,
            margin_y=args.margin_y,
            alignment_size=args.alignment_size,
            margin_top=args.margin_top,
            margin_bottom=args.margin_bottom,
        )
        records: list[DetectionRecord] = list(image_summary.records)
        video_result = None
        if args.video is not None:
            video_result = process_video(
                args.video,
                args.output_dir,
                threshold=args.threshold,
                margin_x=args.margin_x,
                margin_y=args.margin_y,
                alignment_size=args.alignment_size,
                frame_interval=args.frame_interval,
                min_face_width=args.min_face_width,
                min_face_height=args.min_face_height,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
            )
            records.extend(video_result.records)
            print(
                f"Video sampled frames: {video_result.frames_sampled}; "
                f"raw detections: {video_result.raw_detections}; "
                f"filtered detections: {video_result.filtered_detections}; "
                f"saved crops: {video_result.saved_crops}"
            )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    output_dir = args.output_dir
    quality_flag_counts = Counter(
        flag for record in records for flag in record.quality_flags
    )
    write_csv_report(records, report_dir / "detections.csv")
    write_summary(
        report_dir / "summary.txt",
        images_processed=image_summary.images_processed,
        faces_detected=(
            image_summary.faces_detected
            + (video_result.filtered_detections if video_result else 0)
        ),
        images_with_no_faces=image_summary.images_with_no_faces,
        video_frames_sampled=(
            video_result.frames_sampled if video_result is not None else None
        ),
        video_raw_detections=(
            video_result.raw_detections if video_result is not None else 0
        ),
        video_filtered_detections=(
            video_result.filtered_detections if video_result is not None else 0
        ),
        video_saved_crops=(
            video_result.saved_crops if video_result is not None else 0
        ),
        small_faces_count=sum(record.is_small_face for record in records),
        quality_flag_counts=dict(quality_flag_counts),
    )
    print(f"Images processed: {image_summary.images_processed}")
    total_faces = image_summary.faces_detected + (
        video_result.filtered_detections if video_result else 0
    )
    print(f"Faces detected: {total_faces}")
    print(f"Images with no detected faces: {len(image_summary.images_with_no_faces)}")
    print(f"CSV report: {report_dir / 'detections.csv'}")
    if image_summary.errors:
        print(f"Image processing warnings: {len(image_summary.errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
