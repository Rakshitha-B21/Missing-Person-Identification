"""Rank gallery face detections by cosine similarity to a query image."""

from __future__ import annotations

import argparse
import heapq
import importlib
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from resnet15.model import ResNet50
from resnet15.test_checkpoint import (
    load_checkpoint,
    validate_checkpoint_architecture,
    validate_identity_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "resnet15" / "checkpoints" / "resnet50_best_model.pth"
)
RETINAFACE_ROOT = PROJECT_ROOT / "RetinaFace"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def confidence_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a confidence threshold from 0 to 1, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a confidence threshold from 0 to 1, got {value!r}"
        )
    return parsed


def non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number, got {value!r}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search a gallery using ResNet-50 embeddings and cosine similarity."
        )
    )
    parser.add_argument("--query-image", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path)
    parser.add_argument(
        "--video",
        type=Path,
        help="CCTV video to search instead of an image gallery.",
    )
    parser.add_argument(
        "--frame-interval",
        type=positive_int,
        default=30,
        help="Process every Nth video frame (default: 30).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="New directory for video crops and annotated video.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument(
        "--input-mode",
        choices=("cropped", "detect"),
        default="cropped",
        help=(
            "Image-gallery mode: process inputs as crops or detect/crop/align. "
            "Video mode: selects query-image handling; video frames are always "
            "detected (default: cropped)."
        ),
    )
    parser.add_argument(
        "--detection-threshold",
        type=confidence_threshold,
        default=0.80,
        help="RetinaFace confidence threshold (default: 0.80).",
    )
    parser.add_argument(
        "--margin-x",
        type=non_negative_float,
        default=0.30,
        help="Horizontal crop margin used when landmark alignment is unavailable.",
    )
    parser.add_argument(
        "--margin-y",
        type=non_negative_float,
        default=0.30,
        help="Vertical crop margin used when landmark alignment is unavailable.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="Number of face crops to embed at once (default: 32).",
    )
    return parser


def load_retinaface_functions() -> tuple[Callable[..., Any], ...]:
    if not (RETINAFACE_ROOT / "src" / "detector.py").is_file():
        raise FileNotFoundError(
            f"RetinaFace detector was not found under '{RETINAFACE_ROOT}'."
        )

    retinaface_site_packages = (
        RETINAFACE_ROOT / "venv" / "Lib" / "site-packages"
    )
    for path in (RETINAFACE_ROOT, retinaface_site_packages):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        detector = importlib.import_module("src.detector")
        cropper = importlib.import_module("src.cropper")
        aligner = importlib.import_module("src.aligner")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Could not initialize the existing RetinaFace detector, cropper, and "
            "aligner. Check the RetinaFace virtual environment dependencies. "
            f"Original error: {exc}"
        ) from exc

    return (
        detector.detect_faces,
        detector.validate_threshold,
        cropper.crop_face,
        cropper.validate_margins,
        aligner.align_face,
    )


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def preprocess_face(
    image: Any,
    detection: Any,
    transform: transforms.Compose,
    crop_face: Callable[..., Any],
    align_face: Callable[..., Any],
    margin_x: float,
    margin_y: float,
) -> torch.Tensor | None:
    face = extract_face_crop(
        image, detection, crop_face, align_face, margin_x, margin_y
    )
    if face is None:
        return None

    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(rgb_face))


def extract_face_crop(
    image: Any,
    detection: Any,
    crop_face: Callable[..., Any],
    align_face: Callable[..., Any],
    margin_x: float,
    margin_y: float,
) -> Any | None:
    face = align_face(image, detection.landmarks, output_size=112)
    if face is None:
        crop_result = crop_face(
            image,
            detection.bounding_box,
            margin_x=margin_x,
            margin_y=margin_y,
        )
        if crop_result is None:
            return None
        face, _ = crop_result
    return face


def embed_images(model: ResNet50, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        embeddings, _ = model(images)
    if embeddings.ndim != 2 or embeddings.shape[1] != ResNet50.embedding_dim:
        raise RuntimeError(
            f"Expected {ResNet50.embedding_dim}-dimensional embeddings, "
            f"got {tuple(embeddings.shape)}."
        )
    return embeddings


def add_matches(
    query_embeddings: torch.Tensor,
    query_face_numbers: Sequence[int],
    gallery_embeddings: torch.Tensor,
    metadata: Sequence[tuple[Path, int, float, tuple[int, int, int, int]]],
    top_matches: list[tuple[float, str, int, int, float, tuple[int, int, int, int]]],
    top_k: int,
) -> None:
    similarities = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        gallery_embeddings.unsqueeze(0),
        dim=2,
    )
    for gallery_index, (path, face_number, confidence, box) in enumerate(metadata):
        for query_index in range(query_embeddings.shape[0]):
            candidate = (
                float(similarities[query_index, gallery_index].item()),
                str(path),
                face_number,
                query_face_numbers[query_index],
                confidence,
                box,
            )
            if len(top_matches) < top_k:
                heapq.heappush(top_matches, candidate)
            elif candidate[0] > top_matches[0][0]:
                heapq.heapreplace(top_matches, candidate)


def load_model(checkpoint_path: Path) -> ResNet50:
    checkpoint = load_checkpoint(checkpoint_path)
    validate_checkpoint_architecture(checkpoint)
    identity_to_label = validate_identity_mapping(
        checkpoint.get("identity_to_label")
    )
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint is missing a valid model_state_dict.")

    model = ResNet50(num_classes=len(identity_to_label), pretrained=False)
    try:
        model.load_state_dict(model_state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint weights do not match a {len(identity_to_label)}-class "
            f"ResNet-50 model: {exc}"
        ) from exc
    model.eval()
    return model


def load_cropped_image(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    try:
        with Image.open(image_path) as image:
            return transform(image.convert("RGB"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image '{image_path}': {exc}") from exc


def run_cropped_search(
    args: argparse.Namespace,
    model: ResNet50,
    query_path: Path,
    gallery_dir: Path,
) -> int:
    transform = build_transform()
    query_tensor = load_cropped_image(query_path, transform).unsqueeze(0)
    query_embedding = embed_images(model, query_tensor)

    gallery_paths = sorted(
        path
        for path in gallery_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    gallery_paths = [
        path for path in gallery_paths if path.resolve() != query_path
    ]
    if not gallery_paths:
        raise ValueError(
            f"No supported gallery images were found under '{gallery_dir}'."
        )

    top_matches: list[tuple[float, str]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []
    images_processed = 0

    def process_gallery_batch() -> None:
        nonlocal images_processed
        if not batch_tensors:
            return
        embeddings = embed_images(model, torch.stack(batch_tensors))
        similarities = F.cosine_similarity(
            query_embedding.expand_as(embeddings), embeddings, dim=1
        )
        for path, similarity in zip(batch_paths, similarities):
            candidate = (float(similarity.item()), str(path))
            if len(top_matches) < args.top_k:
                heapq.heappush(top_matches, candidate)
            elif candidate[0] > top_matches[0][0]:
                heapq.heapreplace(top_matches, candidate)
        images_processed += len(batch_paths)
        batch_tensors.clear()
        batch_paths.clear()

    for gallery_path in gallery_paths:
        try:
            tensor = load_cropped_image(gallery_path, transform)
        except ValueError as exc:
            print(f"Warning: {exc}; skipping.", file=sys.stderr)
            continue
        batch_tensors.append(tensor)
        batch_paths.append(gallery_path)
        if len(batch_tensors) >= args.batch_size:
            process_gallery_batch()
    process_gallery_batch()

    print(f"Query image: {query_path}")
    print(f"Gallery images processed: {images_processed} of {len(gallery_paths)}")
    if not top_matches:
        print("No readable gallery images were available for comparison.")
        return 0

    print(
        "Top candidates ranked by cosine similarity; scores are not a "
        "calibrated identity decision:"
    )
    for rank, (score, path) in enumerate(
        sorted(top_matches, reverse=True), start=1
    ):
        print(f"{rank}. similarity={score:.6f} image={path}")
    return 0


def run_detect_search(
    args: argparse.Namespace,
    model: ResNet50,
    query_path: Path,
    gallery_dir: Path,
) -> int:
    (
        detect_faces,
        validate_threshold,
        crop_face,
        validate_margins,
        align_face,
    ) = load_retinaface_functions()
    validate_threshold(args.detection_threshold)
    validate_margins(args.margin_x, args.margin_y)

    query_image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if query_image is None:
        raise ValueError(f"Could not read query image: {query_path}")
    query_detections = detect_faces(query_image, args.detection_threshold)
    if not query_detections:
        raise ValueError(
            f"No faces were detected in query image '{query_path}'. "
            "Try a different image or lower --detection-threshold."
        )

    transform = build_transform()
    query_tensors = []
    query_face_numbers = []
    for face_number, detection in enumerate(query_detections, start=1):
        tensor = preprocess_face(
            query_image,
            detection,
            transform,
            crop_face,
            align_face,
            args.margin_x,
            args.margin_y,
        )
        if tensor is not None:
            query_tensors.append(tensor)
            query_face_numbers.append(face_number)
        else:
            print(
                f"Warning: query face {face_number} produced an invalid crop.",
                file=sys.stderr,
            )
    if not query_tensors:
        raise ValueError(f"No valid face crops could be prepared from '{query_path}'.")

    query_embeddings = embed_images(model, torch.stack(query_tensors))
    gallery_paths = sorted(
        path
        for path in gallery_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    gallery_paths = [
        path for path in gallery_paths if path.resolve() != query_path
    ]
    if not gallery_paths:
        raise ValueError(
            f"No supported gallery images were found under '{gallery_dir}'."
        )

    top_matches: list[
        tuple[float, str, int, int, float, tuple[int, int, int, int]]
    ] = []
    batch_tensors: list[torch.Tensor] = []
    batch_metadata: list[
        tuple[Path, int, float, tuple[int, int, int, int]]
    ] = []
    images_read = 0
    faces_detected = 0
    invalid_crops = 0

    def process_gallery_batch() -> None:
        if not batch_tensors:
            return
        gallery_embeddings = embed_images(model, torch.stack(batch_tensors))
        add_matches(
            query_embeddings,
            query_face_numbers,
            gallery_embeddings,
            batch_metadata,
            top_matches,
            args.top_k,
        )
        batch_tensors.clear()
        batch_metadata.clear()

    for gallery_path in gallery_paths:
        gallery_image = cv2.imread(str(gallery_path), cv2.IMREAD_COLOR)
        if gallery_image is None:
            print(
                f"Warning: could not read gallery image '{gallery_path}'; skipping.",
                file=sys.stderr,
            )
            continue
        images_read += 1
        detections = detect_faces(gallery_image, args.detection_threshold)
        faces_detected += len(detections)
        for face_number, detection in enumerate(detections, start=1):
            tensor = preprocess_face(
                gallery_image,
                detection,
                transform,
                crop_face,
                align_face,
                args.margin_x,
                args.margin_y,
            )
            if tensor is None:
                invalid_crops += 1
                continue
            batch_tensors.append(tensor)
            batch_metadata.append(
                (
                    gallery_path,
                    face_number,
                    detection.confidence,
                    detection.bounding_box,
                )
            )
            if len(batch_tensors) >= args.batch_size:
                process_gallery_batch()

    process_gallery_batch()

    print(f"Query image: {query_path}")
    print(f"Query faces embedded: {len(query_tensors)}")
    print(f"Gallery images read: {images_read} of {len(gallery_paths)}")
    print(f"Gallery faces detected: {faces_detected}")
    if invalid_crops:
        print(f"Gallery faces skipped due to invalid crops: {invalid_crops}")
    if not top_matches:
        print("No gallery faces were available for comparison.")
        return 0

    print(
        "Top candidates ranked by cosine similarity; scores are not a "
        "calibrated identity decision:"
    )
    for rank, match in enumerate(sorted(top_matches, reverse=True), start=1):
        score, path, gallery_face, query_face, confidence, box = match
        print(
            f"{rank}. similarity={score:.6f} query_face={query_face} "
            f"gallery_face={gallery_face} detector_confidence={confidence:.3f} "
            f"box={box} image={path}"
        )
    return 0


def format_timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def build_video_query_embeddings(
    args: argparse.Namespace,
    model: ResNet50,
    query_path: Path,
    detect_faces: Callable[..., Any],
    crop_face: Callable[..., Any],
    align_face: Callable[..., Any],
) -> tuple[torch.Tensor, list[int]]:
    transform = build_transform()
    if args.input_mode == "cropped":
        return embed_images(
            model, load_cropped_image(query_path, transform).unsqueeze(0)
        ), [1]

    query_image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if query_image is None:
        raise ValueError(f"Could not read query image: {query_path}")
    detections = detect_faces(query_image, args.detection_threshold)
    if not detections:
        raise ValueError(
            f"No faces were detected in query image '{query_path}'. "
            "Try a different image or use --input-mode cropped."
        )

    query_tensors = []
    query_face_numbers = []
    for face_number, detection in enumerate(detections, start=1):
        tensor = preprocess_face(
            query_image,
            detection,
            transform,
            crop_face,
            align_face,
            args.margin_x,
            args.margin_y,
        )
        if tensor is not None:
            query_tensors.append(tensor)
            query_face_numbers.append(face_number)
    if not query_tensors:
        raise ValueError(f"No valid face crops could be prepared from '{query_path}'.")
    return embed_images(model, torch.stack(query_tensors)), query_face_numbers


def run_video_search(
    args: argparse.Namespace,
    model: ResNet50,
    query_path: Path,
    video_path: Path,
) -> int:
    (
        detect_faces,
        validate_threshold,
        crop_face,
        validate_margins,
        align_face,
    ) = load_retinaface_functions()
    validate_threshold(args.detection_threshold)
    validate_margins(args.margin_x, args.margin_y)

    query_embeddings, query_face_numbers = build_video_query_embeddings(
        args, model, query_path, detect_faces, crop_face, align_face
    )
    transform = build_transform()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    if args.results_dir is None:
        results_dir = (
            PROJECT_ROOT
            / "results"
            / f"video_search_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
    else:
        results_dir = args.results_dir.expanduser().resolve()

    try:
        results_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        capture.release()
        raise FileExistsError(
            f"Results directory already exists; choose a new --results-dir: "
            f"{results_dir}"
        ) from exc
    except OSError:
        capture.release()
        raise

    crops_dir = results_dir / "matching_faces"
    annotated_video_path = results_dir / "annotated_video.mp4"
    writer = None
    top_matches: list[
        tuple[
            float,
            int,
            int,
            str,
            int,
            int,
            float,
            tuple[int, int, int, int],
            Any,
        ]
    ] = []
    frame_number = 0
    sampled_frames = 0
    faces_detected = 0
    match_counter = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_width = 0
    frame_height = 0

    try:
        crops_dir.mkdir()
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1

            height, width = frame.shape[:2]
            if writer is None:
                frame_width, frame_height = width, height
                output_fps = fps if math.isfinite(fps) and fps > 0 else 25.0
                writer = cv2.VideoWriter(
                    str(annotated_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise OSError(
                        f"Could not create annotated video: {annotated_video_path}"
                    )
            elif (width, height) != (frame_width, frame_height):
                raise ValueError(
                    f"Video frame size changed at frame {frame_number}; "
                    "cannot write a consistent annotated video."
                )

            annotated = frame.copy()
            if (frame_number - 1) % args.frame_interval == 0:
                sampled_frames += 1
                detections = detect_faces(frame, args.detection_threshold)
                faces_detected += len(detections)
                face_tensors = []
                face_crops = []
                face_records = []

                for face_number, detection in enumerate(detections, start=1):
                    face_crop = extract_face_crop(
                        frame,
                        detection,
                        crop_face,
                        align_face,
                        args.margin_x,
                        args.margin_y,
                    )
                    if face_crop is None:
                        continue
                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_tensors.append(
                        transform(Image.fromarray(rgb_face))
                    )
                    face_crops.append(face_crop)
                    face_records.append((face_number, detection))

                if face_tensors:
                    frame_embeddings = embed_images(
                        model, torch.stack(face_tensors)
                    )
                    similarities = F.cosine_similarity(
                        query_embeddings.unsqueeze(1),
                        frame_embeddings.unsqueeze(0),
                        dim=2,
                    )
                    best_scores, best_query_indices = similarities.max(dim=0)

                    for face_index, (face_number, detection) in enumerate(
                        face_records
                    ):
                        score = float(best_scores[face_index].item())
                        query_face_number = query_face_numbers[
                            int(best_query_indices[face_index].item())
                        ]
                        x1, y1, x2, y2 = detection.bounding_box
                        top_left = (max(0, x1), max(0, y1))
                        bottom_right = (max(0, x2 - 1), max(0, y2 - 1))
                        cv2.rectangle(
                            annotated, top_left, bottom_right, (0, 255, 0), 2
                        )
                        cv2.putText(
                            annotated,
                            f"similarity {score:.4f}",
                            (top_left[0], max(20, top_left[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        match_counter += 1
                        timestamp_seconds = (
                            (frame_number - 1) / fps
                            if math.isfinite(fps) and fps > 0
                            else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                        )
                        candidate = (
                            score,
                            match_counter,
                            frame_number,
                            format_timestamp(timestamp_seconds),
                            face_number,
                            query_face_number,
                            detection.confidence,
                            detection.bounding_box,
                            face_crops[face_index].copy(),
                        )
                        if len(top_matches) < args.top_k:
                            heapq.heappush(top_matches, candidate)
                        elif score > top_matches[0][0]:
                            heapq.heapreplace(top_matches, candidate)

            writer.write(annotated)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_number == 0:
        raise ValueError(f"No readable frames found in video: {video_path}")

    for rank, match in enumerate(
        sorted(top_matches, key=lambda item: (item[0], item[1]), reverse=True),
        start=1,
    ):
        score, _, matched_frame, timestamp, face_number, _, _, _, crop = match
        crop_path = (
            crops_dir
            / f"frame_{matched_frame:08d}_face_{face_number:03d}_"
            f"similarity_{score:.4f}.jpg"
        )
        if not cv2.imwrite(str(crop_path), crop):
            raise OSError(f"Could not save matching face crop: {crop_path}")
        print(
            f"{rank}. frame={matched_frame} timestamp={timestamp} "
            f"similarity={score:.6f} crop={crop_path}"
        )

    print(f"Video frames read: {frame_number}")
    print(f"Frames sampled: {sampled_frames} (every {args.frame_interval} frame(s))")
    print(f"Faces detected in sampled frames: {faces_detected}")
    print(f"Top matching frames displayed: {len(top_matches)}")
    print(f"Annotated video: {annotated_video_path}")
    print(f"Matching face crops: {crops_dir}")
    print("Similarity ranks candidates only; they are not verified identities.")
    return 0


def run_search(args: argparse.Namespace) -> int:
    query_path = args.query_image.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ResNet-50 checkpoint not found: {checkpoint_path}")
    if (args.gallery_dir is None) == (args.video is None):
        raise ValueError("Provide exactly one of --gallery-dir or --video.")

    model = load_model(checkpoint_path)
    if args.video is not None:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return run_video_search(args, model, query_path, video_path)

    gallery_dir = args.gallery_dir.expanduser().resolve()
    if not gallery_dir.is_dir():
        raise NotADirectoryError(f"Gallery directory not found: {gallery_dir}")
    if args.input_mode == "cropped":
        return run_cropped_search(args, model, query_path, gallery_dir)
    return run_detect_search(args, model, query_path, gallery_dir)


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_search(args)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        cv2.error,
    ) as exc:
        print(f"Search failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
