# AI-Based CCTV Person Identification System — Face Detection Module

This standalone module uses RetinaFace to detect faces, create context-preserving
face crops, align valid five-landmark detections, and annotate input images. It
does **not** identify people or perform tracking.

## How it works

RetinaFace is a single-stage face detector. It predicts face bounding boxes,
confidence scores, and (when available) five facial landmarks in one inference
pass. The confidence is a detector score, not a probability of identity.

The margin-based crop expands the detected box by 30% horizontally and at the
bottom, and 35% at the top by default, then clips it to the image boundary.
These configurable margins add context around the forehead, chin, and sides;
cropping copies source pixels without resizing or changing their detail.

Every detection also gets a lossless PNG crop in `high_resolution_faces/`.
This is the expanded crop taken directly from the decoded source image/video
frame; it is not resized or upscaled. Its saved dimensions depend on the face
box and margins, and cannot add detail that was not present in the source.
The existing JPEG crops remain available in `cropped_faces/`.

The separate aligned output estimates a similarity transform from both eyes,
the nose, and both mouth corners, mapping them to a standard 112×112 template.
Alignment normalizes scale, rotation, and position; this is useful before
embedding extraction and ArcFace-based recognition, but does not recognize a
person. Faces without all valid landmarks still receive margin crops, while
their aligned filename is left empty in the report.

## Installation

Use Python 3.12 and the existing virtual environment, or create one and install
the listed dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The detector uses the `retina-face` package's model and cache. No model is
downloaded or bundled by this project code.

## Run image detection

Put `.jpg`, `.jpeg`, and `.png` images in `images/`. From the project root:

```powershell
.\venv\Scripts\python.exe main.py
```

This processes every supported image recursively under `images/` with a
confidence threshold of `0.80`. Optional settings:

```powershell
.\venv\Scripts\python.exe main.py --threshold 0.80 --margin-x 0.25 --margin-y 0.25 --alignment-size 112
```

## Run video detection later

Video support is optional; no video is required for image processing. The
standalone `process_video` function uses OpenCV `VideoCapture`, samples every
30th frame by default, and writes sampled-frame crops, alignments, annotations,
and report records. The minimum video face-box dimensions default to `0`, so
valid small faces remain available and are quality-flagged instead of being
silently filtered by size. Set either minimum to a positive value to opt into
size filtering; these pixel thresholds depend on decoded video resolution.

Boxes with non-positive dimensions or no intersection with the decoded frame
are discarded. Partially out-of-frame boxes are retained, their crops are
clipped to the frame, and the CSV flags them for review. RetinaFace's installed
detector already applies NMS. This pipeline does not add another NMS pass:
nearby/overlapping candidates can be distinct people, and further suppression
risks dropping real faces. Video
frame numbers are one-based indexes in the original sequential video, not
sample counters. After each successful read, the pipeline uses OpenCV's
`CAP_PROP_POS_FRAMES` next-frame position minus one as the current zero-based
frame position (falling back to the sequential read count when that property is
unavailable). Timestamps are calculated from that original frame position and
the video FPS; when FPS is missing or invalid, timestamps are left empty.
Earlier outputs are never removed. If output files for the same video already
exist, processing stops with an explicit error; use a separate output directory
to avoid replacing those results.

From the project root, an optional CLI run is:

```powershell
.\venv\Scripts\python.exe main.py --video videos/test_video.mp4 --frame-interval 10
```

When `--video` is supplied, the CLI processes `images/` as usual and then
processes the video. A missing or unreadable video produces an explicit error.
The CSV records the video name, one-based sampled frame number, timestamp when
frame rate is available, face ID within that frame, confidence, box, and saved
crop/alignment filenames, original face-box width and height in pixels,
Laplacian-variance blur score, grayscale brightness and contrast, and quality
flags. Small faces (either dimension under 32 pixels), blurry faces (Laplacian
variance below 1,000), under/overexposed faces (mean brightness below 50 or
above 205), low-contrast faces (standard deviation below 15), partially cropped
faces, and unavailable alignments are flagged for review. These are heuristic
review indicators, not accuracy classifications or automatic rejection rules.
Face IDs restart in each sampled frame and are not persistent person
identities. The run summary reports sampled frames, raw RetinaFace result
entries, detections retained after configured filtering, saved crops, and
quality-flag counts. For example, to opt into minimum-size filtering:

```powershell
.\venv\Scripts\python.exe main.py --video videos/test_video.mp4 --frame-interval 10 --min-face-width 16 --min-face-height 20
```

Set either `--min-face-width` or `--min-face-height` to `0` to disable that
dimension's minimum. This is frame-level detection only, not tracking.

## Output folders

- `output/cropped_faces/`: original-resolution face crops with configurable
  margins, saved as high-quality JPEGs.
- `output/high_resolution_faces/`: lossless PNG crops directly sliced from
  source images or decoded video frames, with configurable margins and no
  resizing.
- `output/aligned_faces/`: separate, square, landmark-aligned images.
- `output/annotated_images/`: source images or sampled video frames labeled
  with the face ID, confidence score, and original bounding box.
- `output/verification/`: optional paginated face-crop and aligned-face contact
  sheets for visual review. Generate them from existing output directories with
  `src.contact_sheet.create_contact_sheets`; empty input directories produce a
  sheet labeled "No faces found".
- `output/reports/detections.csv`: per-detection image/video report.
- `output/reports/summary.txt`: processed-image, detected-face, and no-face
  image totals, including the no-face image names; video runs also include
  sampled-frame, raw/filtered-detection, saved-crop totals, and quality-flag
  counts. CSV quality measurements are made on the original detected face box
  after clipping the measurement region to the source frame.

The report's bounding-box coordinates and width/height are the detector's
original face-box measurements; crop filenames refer to the expanded crops.
Aligned crops remain 112×112 by default. Face IDs are local to an image or
video frame and are not persistent identity labels.

## Limitations and future integration

Detection confidence does not establish a person's identity. Results depend on
image quality, pose, lighting, occlusion, and the detector's model. Quality
thresholds are practical review heuristics and have not been calibrated against
ground-truth labels; inspect crops and annotations before drawing accuracy
conclusions.
Aligned crops can later be passed to a ResNet-18 embedding model trained with
ArcFace loss and compared with cosine similarity in the separate identification
module. This project intentionally does not implement that recognition,
database, or cross-camera tracking functionality.
