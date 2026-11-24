# Missing Person Identification

## Project overview

This project is a prototype for finding candidate appearances of a person in
face images or sampled CCTV video. A query face is converted into a ResNet-50
embedding, detected faces in a gallery or video are embedded the same way, and
cosine similarity is used to rank candidates.

The output is a ranked list for human review. A similarity score does **not**
verify a person's identity, and this prototype does not provide persistent
tracking across cameras.

## Problem statement

Surveillance imagery can contain low-resolution faces, varied pose and
lighting, motion blur, and occlusion. The project explores a pipeline that
combines face detection and alignment with learned face embeddings to search a
gallery or sampled video frames for appearances similar to a supplied query.

The project is an engineering prototype and evaluation exercise, not a
production missing-person service or a validated identification system.

## Objectives

- Fine-tune an ImageNet-pretrained ResNet-50 identity classifier that also
  produces normalized 512-dimensional face embeddings.
- Reuse RetinaFace for face detection and the project's crop/alignment code for
  detected faces.
- Search already-cropped face images directly, or search faces detected in a
  video.
- Evaluate an identity-matched pilot using gallery, mated-probe, and
  unmated-probe data.
- Report candidate rankings and exploratory metrics without claiming identity
  verification.

## Technologies

- Python 3.12
- PyTorch and Torchvision
- Pillow for image loading
- OpenCV for video I/O and image operations
- RetinaFace for face detection
- NumPy and SciPy for evaluation data and distances
- Matplotlib for similarity-distribution plots

The RetinaFace subproject's `requirements.txt` lists `retina-face`,
`opencv-python`, and `numpy`. The ResNet and evaluation scripts additionally
use PyTorch, Torchvision, Pillow, and SciPy; the plotting script uses
Matplotlib.

## System architecture and workflow

```text
Training images
      |
      v
Resize to 112x112 -> ResNet-50 -> identity classification loss
                                  |
                                  +-> normalized 512-D face embeddings

Query face ------------------------------+
                                          |
Gallery images -> optional RetinaFace ----+-> ResNet-50 embeddings
Video -> sampled frames -> RetinaFace ----+-> cosine-similarity ranking
                                               |
                                               +-> top candidate images/frames
                                               +-> saved crops and annotated video

Official pilot gallery/mated/unmated splits
      -> shared-identity feature extraction
      -> per-identity gallery templates
      -> nearest-identity distances and open-set metrics
```

### Image and video search

`run_missing_person_search.py` supports two image-gallery input modes:

- `cropped` (default): reads each image as an RGB face crop using Pillow, then
  applies the ResNet preprocessing directly.
- `detect`: runs the existing RetinaFace detector on the query and gallery
  images. Detected faces are aligned from five landmarks when available, with a
  margin crop as fallback.

For video search, the script samples frames at `--frame-interval`, applies
RetinaFace to sampled frames, embeds detected faces, and ranks them against the
query embedding. It writes the annotated video and top-ranked face crops into a
new results directory. The query image is treated as an already-cropped face by
default; `--input-mode detect` can be used if the query is a full image.

The standalone RetinaFace tools also support image/video detection, cropping,
alignment, quality flags, annotations, and reports. Those outputs are
preprocessing and review aids; they are not identity decisions.

### Training and preprocessing

Training and evaluation images are resized to 112 x 112 pixels, converted to
tensors, and normalized with mean and standard deviation 0.5 per RGB channel.
Training runs on CPU and uses an identity-wise 80/20 train/validation split.

## ResNet-50 face model

The model is implemented in `resnet15/model.py` using
`torchvision.models.resnet50` and `ResNet50_Weights.DEFAULT` when its ImageNet
weights are available. If weights cannot be retrieved, initialization falls
back to random weights with a warning. It contains:

- The standard ResNet-50 backbone with its ImageNet classification head removed.
- A 2048-to-512 linear embedding projection.
- Batch normalization and L2 normalization of the 512-dimensional embedding.
- A linear identity-classification head sized to the selected training
  identities.

## Dataset

The project uses the QMUL-SurvFace dataset in the local `dataset/QMUL-SurvFace`
directory:

- `training_set` contains identity-organized training images. The training code
  expects 5,319 identity folders in the full training split.
- `Face_Identification_Test_Set` provides gallery, mated-probe, and
  unmated-probe metadata and images for identification evaluation.

The dataset is not reproduced in this documentation. Follow its license,
access, and privacy terms when using the data.

## Training run and checkpoint

New ResNet-50 training saves to
`resnet15/checkpoints/resnet50_best_model.pth`. The prior ResNet-15 checkpoint
at `resnet15/checkpoints/best_model.pth` and the
`resnet15/model_resnet15_backup.py` backup are left unchanged and are not
compatible with the ResNet-50 consumers. ResNet-50 checkpoints include an
architecture marker so older checkpoints are rejected explicitly.

The training script uses cross-entropy loss and AdamW, with a default of 10
epochs and a CPU-safe default batch size of 32. It supports seeded identity
selection and optional per-identity and total-image caps. Resume with
`--resume` using the same identity selection; it restores model and optimizer
state only from the ResNet-50 checkpoint.

## Pilot open-set evaluation

The historical ResNet-15 pilot remains in
`resnet15/official_results/pilot_matched_100/`. New ResNet-50 feature
extraction writes separately to
`resnet15/official_results/resnet50_pilot_matched_100/`.
It selects 100 identities shared by the gallery and mated-probe sets, includes
their available images, and uses 1,000 unmated probes. The generated artifacts
contain:

| Split / statistic | Count |
| --- | ---: |
| Selected identities | 100 |
| Gallery images | 1,435 |
| Gallery templates | 100 |
| Mated probes | 1,655 |
| Unmated probes | 1,000 |
| Gallery/mated identity overlap | 100% (100/100) |

Gallery templates are formed by averaging embeddings for each identity and
normalizing the resulting template. The evaluator computes Euclidean distances
to these templates, uses the nearest identity, and constructs an open-set
identification curve from mated and unmated probe distances.

| Pilot metric | Result |
| --- | ---: |
| Rank-1 mated probes correctly identified | 0.0677 |
| TPIR at FPIR 0.1 | 0.0079 |
| TPIR at FPIR 0.2 | 0.0278 |
| TPIR at FPIR 0.3 | 0.0393 |
| Open-set identification AUC | 0.0456 |

**This is a 100-identity pilot, not the full official QMUL-SurvFace
benchmark.** Its results should be treated as exploratory and are not directly
generalizable to the full protocol.

## CCTV demonstration

The recorded demonstration reports **677 video frames read**, **46 frames
sampled**, **209 faces detected**, and **5 top candidate matches generated**.
The existing `final_cctv_results/` folder contains an annotated video and five
candidate face crops. The saved video metadata reports 677 frames; the sampled
frame and detection totals are reported run results.

Candidates are ranked by cosine similarity. These scores do not establish that
any candidate is the queried person or a verified identity.

## Windows PowerShell setup and execution

Run commands from the project root. Use the existing Python 3.12 RetinaFace
virtual environment so the detector and search dependencies can share one
interpreter. Create it only if it is missing:

```powershell
if (-not (Test-Path .\RetinaFace\venv\Scripts\python.exe)) {
    py -3.12 -m venv RetinaFace\venv
}
.\RetinaFace\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r RetinaFace\requirements.txt
python -m pip install torch torchvision pillow scipy matplotlib tf-keras
```

The `tf-keras` package is needed with TensorFlow/Keras versions that trigger the
RetinaFace compatibility check. If PyTorch is already installed in a suitable
Python 3.12 environment, use that environment instead of reinstalling it.

### Train or inspect the model

For full training with the default 5,319 identities, use a smaller batch size
to limit memory use:

```powershell
python resnet15\train.py --batch-size 8
```

Training writes only `resnet15\checkpoints\resnet50_best_model.pth`; the
existing ResNet-15 checkpoints are not overwritten. A smaller seeded subset
can be used for a quick pilot with
`--max-identities 100 --max-total-images 563 --seed 42`. Resume a compatible
run with the same selection arguments and `--resume` (increase `--epochs` to
train beyond the epoch count of the saved checkpoint):

```powershell
python resnet15\train.py --batch-size 8 --epochs 20 --resume
```

Load the saved checkpoint and run one-image inference:

```powershell
python resnet15\test_checkpoint.py
python resnet15\evaluate_matching.py
python resnet15\test_matching.py
```

### Search a cropped-image gallery

```powershell
python run_missing_person_search.py `
  --query-image "path\to\query_face.jpg" `
  --gallery-dir "path\to\gallery" `
  --input-mode cropped `
  --top-k 10
```

Use `--input-mode detect` when the query and gallery images are full images that
need face detection.

### Search a CCTV video

```powershell
python run_missing_person_search.py `
  --query-image "path\to\query_face.jpg" `
  --video "path\to\cctv_video.mp4" `
  --frame-interval 15 `
  --top-k 5
```

Video frames are always processed with RetinaFace. The query defaults to
cropped-face mode; add `--input-mode detect` for a full-image query. The script
creates a timestamped results folder unless `--results-dir` is supplied. An
explicit results directory must not already exist.

### Extract and evaluate the matched-identity pilot

```powershell
python resnet15\official_evaluation.py --batch-size 8
python resnet15\evaluate_official_pilot.py
```

Feature extraction writes to
`resnet15\official_results\resnet50_pilot_matched_100\` and refuses to
overwrite an existing folder. The evaluation uses that folder.

## Limitations and future scope

### Limitations

- The historical ResNet-15 checkpoint was trained on a small selected subset
  and had a 31.03% best validation accuracy; this is not a ResNet-50 result.
- The evaluation covers 100 identities only and is not the complete official
  QMUL-SurvFace benchmark.
- Face quality, camera conditions, pose, illumination, resolution, and
  occlusion can substantially affect detection and embeddings.
- The system ranks candidates but does not verify identities, calibrate an
  operational matching threshold, or track people persistently.
- The current pipeline has not been validated for operational surveillance or
  consequential decisions.

### Future enhancements

- Train and evaluate with broader identity and image coverage under the full
  benchmark protocol.
- Explore metric-learning objectives, stronger augmentation, and more robust
  face alignment.
- Develop a validation-only process for threshold calibration and report
  uncertainty and performance across operating conditions.
- Add optional tracking and carefully evaluated video-level aggregation.
- Improve reproducibility with saved run configurations and complete evaluation
  manifests.
- Apply privacy-by-design practices, access controls, retention limits, and
  human review before any real-world use.

## Responsible-use note

Face embeddings and face images are sensitive biometric data. Use this project
only with appropriate authorization and safeguards. Similarity rankings are
investigative candidates for review, not proof of identity.
