from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.aligner import align_face
from src.contact_sheet import _image_label, create_contact_sheets
from src.cropper import crop_face
from src.detector import FaceDetection, detect_faces
from src.image_processor import process_images
from src.quality import measure_face_quality
from src.reporter import DetectionRecord, write_csv_report
from src.video_processor import filter_video_detections, process_video


def _raw_face(x_offset: int = 0, confidence: float = 0.95) -> dict:
    return {
        "facial_area": [
            20 + x_offset,
            15,
            60 + x_offset,
            70,
        ],
        "score": confidence,
        "landmarks": {
            "left_eye": [48 + x_offset, 32],
            "right_eye": [32 + x_offset, 32],
            "nose": [40 + x_offset, 45],
            "mouth_left": [48 + x_offset, 56],
            "mouth_right": [32 + x_offset, 56],
        },
    }


def _face_detection(x_offset: int = 0) -> FaceDetection:
    return FaceDetection(
        bounding_box=(20 + x_offset, 15, 60 + x_offset, 70),
        confidence=0.95,
        landmarks={
            "left_eye": (48 + x_offset, 32),
            "right_eye": (32 + x_offset, 32),
            "nose": (40 + x_offset, 45),
            "mouth_left": (48 + x_offset, 56),
            "mouth_right": (32 + x_offset, 56),
        },
    )


def _detection_with_box(box: tuple[int, int, int, int]) -> FaceDetection:
    detection = _face_detection()
    return FaceDetection(box, detection.confidence, detection.landmarks)


class CropperTests(unittest.TestCase):
    def test_expands_crop_and_clamps_to_image(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        crop, coordinates = crop_face(image, (10, 10, 50, 70))
        self.assertEqual(coordinates, (0, 0, 62, 88))
        self.assertEqual(crop.shape, (88, 62, 3))

    def test_rejects_invalid_box(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        self.assertIsNone(crop_face(image, (20, 20, 10, 50)))


class DetectorTests(unittest.TestCase):
    @patch("src.detector.RetinaFace.detect_faces")
    def test_filters_threshold_and_keeps_multiple_faces(self, detect_mock) -> None:
        detect_mock.return_value = {
            "face_1": _raw_face(),
            "face_2": _raw_face(45, confidence=0.85),
            "face_3": _raw_face(80, confidence=0.79),
        }
        detections = detect_faces(np.zeros((100, 140, 3), dtype=np.uint8), 0.80)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].bounding_box, (20, 15, 60, 70))
        self.assertEqual(len(detections[0].landmarks), 5)

    @patch("src.detector.RetinaFace.detect_faces", return_value=False)
    def test_no_faces_is_an_empty_list(self, _detect_mock) -> None:
        self.assertEqual(detect_faces(np.zeros((20, 20, 3), dtype=np.uint8)), [])
        self.assertEqual(_detect_mock.call_count, 1)


class AlignmentTests(unittest.TestCase):
    def test_alignment_has_requested_dimensions(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        aligned = align_face(image, _face_detection().landmarks, output_size=112)
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.shape, (112, 112, 3))

    def test_missing_landmarks_are_gracefully_skipped(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(align_face(image, None))

    def test_degenerate_or_out_of_frame_landmarks_are_skipped(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        degenerate = {name: (25.0, 25.0) for name in _face_detection().landmarks}
        out_of_frame = dict(_face_detection().landmarks)
        out_of_frame["nose"] = (120.0, 40.0)
        self.assertIsNone(align_face(image, degenerate))
        self.assertIsNone(align_face(image, out_of_frame))


class ImageProcessorTests(unittest.TestCase):
    @patch("src.image_processor.detect_faces")
    def test_processes_multiple_faces_and_no_face_image(self, detect_mock) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            frame = np.full((100, 140, 3), 127, dtype=np.uint8)
            cv2.imwrite(str(image_dir / "multiple.jpg"), frame)
            cv2.imwrite(str(image_dir / "empty.png"), frame)
            detect_mock.side_effect = [[], [_face_detection(), _face_detection(65)]]

            summary = process_images(image_dir, root / "output")

            self.assertEqual(summary.images_processed, 2)
            self.assertEqual(summary.faces_detected, 2)
            self.assertEqual(summary.images_with_no_faces, ["empty.png"])
            self.assertEqual(detect_mock.call_count, 2)
            self.assertEqual(len(list((root / "output" / "cropped_faces").glob("*.jpg"))), 2)
            high_resolution_faces = list(
                (root / "output" / "high_resolution_faces").glob("*.png")
            )
            self.assertEqual(len(high_resolution_faces), 2)
            high_resolution_path = (
                root
                / "output"
                / "high_resolution_faces"
                / "multiple.jpg_face_001.png"
            )
            high_resolution = cv2.imread(str(high_resolution_path))
            self.assertEqual(high_resolution.shape, (86, 64, 3))
            source = cv2.imread(str(image_dir / "multiple.jpg"))
            expected_crop, _ = crop_face(
                source,
                _face_detection().bounding_box,
            )
            self.assertTrue(np.array_equal(high_resolution, expected_crop))
            self.assertEqual(len(list((root / "output" / "aligned_faces").glob("*.jpg"))), 2)
            aligned = cv2.imread(
                str(root / "output" / "aligned_faces" / "multiple.jpg_face_001.jpg")
            )
            self.assertEqual(aligned.shape, (112, 112, 3))
            self.assertEqual(summary.records[0].original_face_width, 40)
            self.assertEqual(summary.records[0].original_face_height, 55)
            self.assertEqual(
                summary.records[0].high_resolution_crop_filename,
                "high_resolution_faces/multiple.jpg_face_001.png",
            )
            self.assertTrue(
                (root / "output" / "annotated_images" / "empty.png_annotated.jpg").is_file()
            )

    @patch("src.image_processor.detect_faces", return_value=[_face_detection()])
    def test_existing_image_outputs_are_not_overwritten(self, _detect_mock) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            cv2.imwrite(
                str(image_dir / "face.jpg"),
                np.full((100, 140, 3), 127, dtype=np.uint8),
            )
            previous_crop = (
                root / "output" / "high_resolution_faces" / "face.jpg_face_001.png"
            )
            previous_crop.parent.mkdir(parents=True)
            previous_crop.write_bytes(b"preserve")

            with self.assertRaisesRegex(FileExistsError, "separate output directory"):
                process_images(image_dir, root / "output")

            self.assertEqual(previous_crop.read_bytes(), b"preserve")


class QualityTests(unittest.TestCase):
    def test_small_dark_low_contrast_face_is_flagged_not_discarded(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        metrics = measure_face_quality(image, (10, 10, 30, 34))
        self.assertEqual(metrics.blur_score, 0.0)
        self.assertEqual(metrics.brightness_mean, 0.0)
        self.assertEqual(
            metrics.flags,
            ("small_face", "blurry", "underexposed", "low_contrast"),
        )

    def test_partially_out_of_frame_face_is_measured_and_flagged(self) -> None:
        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        metrics = measure_face_quality(image, (-4, 10, 20, 38))
        self.assertIn("partially_cropped", metrics.flags)
        self.assertIsNotNone(metrics.brightness_mean)


class ReporterTests(unittest.TestCase):
    def test_reports_original_dimensions_and_flags_small_faces(self) -> None:
        record = DetectionRecord(
            media_name="small-face.jpg",
            frame_number=None,
            timestamp="",
            face_id="face_001",
            confidence_score=0.95,
            bounding_box=(10, 20, 26, 40),
            crop_filename="cropped_faces/small-face.jpg_face_001.jpg",
            aligned_filename="aligned_faces/small-face.jpg_face_001.jpg",
            high_resolution_crop_filename=(
                "high_resolution_faces/small-face.jpg_face_001.png"
            ),
        )
        self.assertEqual(record.original_face_width, 16)
        self.assertEqual(record.original_face_height, 20)
        self.assertTrue(record.is_small_face)

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "detections.csv"
            write_csv_report([record], report_path)
            with report_path.open(encoding="utf-8", newline="") as report_file:
                row = next(csv.DictReader(report_file))

            self.assertEqual(row["original_face_width"], "16")
            self.assertEqual(row["original_face_height"], "20")
            self.assertEqual(
                row["high_resolution_crop_filename"],
                "high_resolution_faces/small-face.jpg_face_001.png",
            )
            self.assertEqual(row["small_face"], "True")


class ContactSheetTests(unittest.TestCase):
    def test_labels_include_face_id_and_source_frame(self) -> None:
        self.assertEqual(
            _image_label(Path("cctv_test.mp4_frame_000031_face_002.jpg")),
            "Frame 000031 | face_002",
        )
        self.assertEqual(
            _image_label(Path("test.jpg_face_001.jpg")),
            "face_001",
        )

    def test_creates_a_valid_sheet_when_there_are_no_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "cropped_faces"
            images_dir.mkdir()

            sheets = create_contact_sheets(
                images_dir,
                root / "verification",
                "cropped_faces",
            )

            self.assertEqual(len(sheets), 1)
            sheet = cv2.imread(str(sheets[0]))
            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.shape[1], 6 * 180)


class VideoProcessorTests(unittest.TestCase):
    def test_filters_configured_tiny_boxes_but_keeps_partial_faces(self) -> None:
        detections = [
            _detection_with_box((10, 10, 25, 29)),  # 15 x 19: below configured minima
            _detection_with_box((30, 10, 46, 30)),  # 16 x 20: at configured minima
            _detection_with_box((-1, 10, 30, 40)),
            _detection_with_box((10, -1, 30, 40)),
            _detection_with_box((90, 10, 110, 40)),
            _detection_with_box((10, 80, 30, 101)),
            _detection_with_box((30, 30, 20, 40)),
            _detection_with_box((30, 30, 30, 40)),
        ]
        kept = filter_video_detections(
            detections,
            frame_width=100,
            frame_height=100,
            min_face_width=16,
            min_face_height=20,
        )
        self.assertEqual(
            [item.bounding_box for item in kept],
            [
                (30, 10, 46, 30),
                (-1, 10, 30, 40),
                (10, -1, 30, 40),
                (90, 10, 110, 40),
                (10, 80, 30, 101),
            ],
        )

    def test_default_filter_keeps_small_and_partially_visible_faces(self) -> None:
        detections = [
            _detection_with_box((5, 5, 12, 12)),
            _detection_with_box((-2, 5, 12, 24)),
            _detection_with_box((101, 5, 120, 24)),
        ]
        self.assertEqual(
            filter_video_detections(detections, frame_width=100, frame_height=100),
            detections[:2],
        )

    def test_overlapping_detections_are_not_suppressed_a_second_time(self) -> None:
        detections = [
            _detection_with_box((10, 10, 30, 30)),
            _detection_with_box((22, 10, 42, 30)),
        ]
        self.assertEqual(
            filter_video_detections(detections, frame_width=100, frame_height=100),
            detections,
        )

    @patch(
        "src.video_processor.detect_faces_with_raw_count",
        return_value=(
            [
                _face_detection(),
                _detection_with_box((60, 15, 95, 70)),
                _detection_with_box((110, 10, 130, 40)),
            ],
            3,
        ),
    )
    @patch("src.video_processor.cv2.VideoCapture")
    def test_samples_frames_and_records_video_metadata(
        self, capture_class, detect_mock
    ) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.frames = [
                    np.full((100, 100, 3), index, dtype=np.uint8)
                    for index in range(5)
                ]
                self.read_count = 0
                self.released = False

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                if property_id == cv2.CAP_PROP_FPS:
                    return 2.0
                if property_id == cv2.CAP_PROP_POS_FRAMES:
                    return float(self.read_count)
                return 0.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.read_count >= len(self.frames):
                    return False, None
                frame = self.frames[self.read_count]
                self.read_count += 1
                return True, frame

            def release(self) -> None:
                self.released = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.mp4"
            video_path.touch()
            old_crop = (
                root
                / "output"
                / "cropped_faces"
                / "previous.mp4_frame_000099_face_001.jpg"
            )
            old_crop.parent.mkdir(parents=True)
            old_crop.write_bytes(b"old video output")
            capture = FakeCapture()
            capture_class.return_value = capture

            result = process_video(video_path, root / "output", frame_interval=2)

            self.assertEqual(result.frames_sampled, 3)
            self.assertEqual(result.raw_detections, 9)
            self.assertEqual(result.filtered_detections, 6)
            self.assertEqual(result.faces_detected, 6)
            self.assertEqual(result.saved_crops, 6)
            self.assertEqual(
                [row.frame_number for row in result.records],
                [1, 1, 3, 3, 5, 5],
            )
            self.assertEqual(
                [row.timestamp for row in result.records],
                [
                    "00:00:00.000",
                    "00:00:00.000",
                    "00:00:01.000",
                    "00:00:01.000",
                    "00:00:02.000",
                    "00:00:02.000",
                ],
            )
            report_path = root / "output" / "reports" / "detections.csv"
            write_csv_report(result.records, report_path)
            with report_path.open(encoding="utf-8", newline="") as report_file:
                csv_records = list(csv.DictReader(report_file))
            self.assertEqual(
                [record["frame_number"] for record in csv_records],
                ["1", "1", "3", "3", "5", "5"],
            )
            self.assertEqual(
                [record["timestamp"] for record in csv_records],
                [record.timestamp for record in result.records],
            )
            self.assertTrue(
                all(record["high_resolution_crop_filename"].endswith(".png")
                    for record in csv_records)
            )
            self.assertEqual(detect_mock.call_count, 3)
            self.assertTrue(capture.released)
            self.assertTrue(old_crop.exists())
            self.assertEqual(
                len(list((root / "output" / "cropped_faces").glob("*.jpg"))),
                result.saved_crops + 1,
            )
            high_resolution_path = (
                root
                / "output"
                / "high_resolution_faces"
                / "sample.mp4_frame_000001_face_001.png"
            )
            high_resolution = cv2.imread(str(high_resolution_path))
            self.assertEqual(high_resolution.shape, (86, 64, 3))
            self.assertEqual(
                len(
                    list(
                        (root / "output" / "high_resolution_faces").glob("*.png")
                    )
                ),
                result.saved_crops,
            )

    @patch(
        "src.video_processor.detect_faces_with_raw_count",
        return_value=([_face_detection()], 1),
    )
    @patch("src.video_processor.cv2.VideoCapture")
    def test_invalid_fps_leaves_timestamp_empty(
        self, capture_class, _detect_mock
    ) -> None:
        class FakeCapture:
            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                if property_id == cv2.CAP_PROP_FPS:
                    return 0.0
                if property_id == cv2.CAP_PROP_POS_FRAMES:
                    return 1.0
                return 0.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if hasattr(self, "read_once"):
                    return False, None
                self.read_once = True
                return True, np.full((100, 100, 3), 127, dtype=np.uint8)

            def release(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "invalid-fps.mp4"
            video_path.touch()
            capture_class.return_value = FakeCapture()

            result = process_video(video_path, root / "output", frame_interval=1)

            self.assertEqual([record.timestamp for record in result.records], [""])

    @patch(
        "src.video_processor.detect_faces_with_raw_count",
        return_value=([_face_detection()], 1),
    )
    @patch("src.video_processor.cv2.VideoCapture")
    def test_uses_reported_original_frame_position(
        self, capture_class, _detect_mock
    ) -> None:
        class FakeCapture:
            read_count = 0

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                if property_id == cv2.CAP_PROP_FPS:
                    return 2.0
                if property_id == cv2.CAP_PROP_POS_FRAMES:
                    return float(self.read_count + 4)
                return 0.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.read_count >= 3:
                    return False, None
                self.read_count += 1
                return True, np.full((100, 100, 3), 127, dtype=np.uint8)

            def release(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "offset-position.mp4"
            video_path.touch()
            capture_class.return_value = FakeCapture()

            result = process_video(video_path, root / "output", frame_interval=2)

            self.assertEqual([record.frame_number for record in result.records], [5, 7])
            self.assertEqual(
                [record.timestamp for record in result.records],
                ["00:00:02.000", "00:00:03.000"],
            )

    @patch("src.video_processor.cv2.VideoCapture")
    def test_unreadable_video_is_reported(self, capture_class) -> None:
        class FakeCapture:
            def isOpened(self) -> bool:
                return False

            def release(self) -> None:
                self.released = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "unreadable.mp4"
            video_path.touch()
            capture = FakeCapture()
            capture_class.return_value = capture

            with self.assertRaisesRegex(OSError, "Could not open video"):
                process_video(video_path, Path(temporary_directory) / "output")

            self.assertTrue(capture.released)

    @patch("src.video_processor.cv2.VideoCapture")
    def test_existing_outputs_for_video_are_never_overwritten(
        self, capture_class
    ) -> None:
        class FakeCapture:
            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                self.released = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.mp4"
            video_path.touch()
            existing_output = (
                root
                / "output"
                / "high_resolution_faces"
                / "sample.mp4_frame_000001_face_001.png"
            )
            existing_output.parent.mkdir(parents=True)
            existing_output.write_bytes(b"keep this result")
            capture = FakeCapture()
            capture_class.return_value = capture

            with self.assertRaisesRegex(FileExistsError, "separate output directory"):
                process_video(video_path, root / "output")

            self.assertEqual(existing_output.read_bytes(), b"keep this result")
            self.assertTrue(capture.released)

    def test_missing_video_path_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            process_video(Path("missing-video.mp4"), Path("output"))


if __name__ == "__main__":
    unittest.main()
