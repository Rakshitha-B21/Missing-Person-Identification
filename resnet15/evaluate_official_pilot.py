"""Evaluate the matched-identity SurvFace pilot using open-set identification."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


BASE = (
    Path(__file__).resolve().parent
    / "official_results"
    / "resnet50_pilot_matched_100"
)
EMBEDDING_DIM = 512


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_path = BASE / f"{name}_features.npy"
    ids_path = BASE / f"{name}_ids.npy"
    filenames_path = BASE / f"{name}_filenames.txt"
    for path in (feature_path, ids_path, filenames_path):
        if not path.is_file():
            raise FileNotFoundError(f"Pilot result file not found: {path}")

    features = np.load(feature_path, mmap_mode="r")
    identity_ids = np.load(ids_path)
    with filenames_path.open("r", encoding="utf-8") as file:
        filenames = [line.rstrip("\r\n") for line in file]

    if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"{name} features must have shape (N, {EMBEDDING_DIM})."
        )
    if features.shape[0] != len(identity_ids) or features.shape[0] != len(filenames):
        raise ValueError(
            f"{name} feature, identity, and filename counts do not match."
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{name} features contain non-finite values.")
    if len(features) == 0:
        raise ValueError(f"{name} set is empty.")
    return np.asarray(features, dtype=np.float32), identity_ids, filenames


def make_gallery_templates(
    gallery_features: np.ndarray, gallery_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(gallery_ids)
    if unique_ids.size == 0:
        raise ValueError("Gallery contains no identities.")

    templates = np.stack(
        [
            gallery_features[gallery_ids == identity].mean(axis=0)
            for identity in unique_ids
        ]
    )
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("A gallery identity template has zero or invalid norm.")
    templates = templates / norms
    return unique_ids, templates.astype(np.float32, copy=False)


def make_open_set_curve(
    mated_min_distances: np.ndarray,
    mated_correct: np.ndarray,
    unmated_min_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if mated_min_distances.size == 0:
        raise ValueError("No mated probes are available for evaluation.")
    if unmated_min_distances.size == 0:
        raise ValueError(
            "No unmated probes are available; FPIR and open-set AUC cannot be computed."
        )
    if not np.isfinite(mated_min_distances).all() or not np.isfinite(
        unmated_min_distances
    ).all():
        raise ValueError("Nearest-identity distances contain non-finite values.")

    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([-np.inf]),
                mated_min_distances,
                unmated_min_distances,
                np.asarray([np.inf]),
            )
        )
    )
    fpir_values = np.empty(len(thresholds), dtype=np.float64)
    tpir_values = np.empty(len(thresholds), dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        fpir_values[index] = np.mean(unmated_min_distances <= threshold)
        tpir_values[index] = np.mean(
            mated_correct & (mated_min_distances <= threshold)
        )

    unique_fpir = np.unique(fpir_values)
    max_tpir_at_fpir = np.asarray(
        [tpir_values[fpir_values == fpir].max() for fpir in unique_fpir],
        dtype=np.float64,
    )
    return unique_fpir, max_tpir_at_fpir


def evaluate() -> None:
    selected_ids_path = BASE / "selected_identity_ids.npy"
    if not selected_ids_path.is_file():
        raise FileNotFoundError(f"Selected identity list not found: {selected_ids_path}")
    selected_ids = np.load(selected_ids_path)
    if selected_ids.ndim != 1 or selected_ids.size == 0:
        raise ValueError("Selected identity list must be a non-empty 1-D array.")

    gallery_features, gallery_ids, gallery_names = load_split("gallery")
    mated_features, mated_ids, mated_names = load_split("mated_probe")
    unmated_features, unmated_ids, unmated_names = load_split("unmated_probe")

    if not np.all(unmated_ids == -100):
        raise ValueError("Unmated probes must use the sentinel identity ID -100.")
    gallery_identity_ids, templates = make_gallery_templates(
        gallery_features, gallery_ids
    )
    if not np.array_equal(np.sort(np.unique(selected_ids)), gallery_identity_ids):
        raise ValueError(
            "Selected identity list does not match the identities in the gallery."
        )
    if not np.array_equal(np.sort(np.unique(selected_ids)), np.unique(mated_ids)):
        raise ValueError(
            "Selected identity list does not match the identities in mated probes."
        )
    missing_identities = np.setdiff1d(np.unique(mated_ids), gallery_identity_ids)
    if missing_identities.size:
        preview = ", ".join(str(identity) for identity in missing_identities[:10])
        raise ValueError(
            f"{missing_identities.size} mated-probe identities have no gallery "
            f"template; evaluation stopped. Missing IDs: {preview}"
        )

    overlap = np.intersect1d(gallery_identity_ids, np.unique(mated_ids))
    overlap_percent = 100.0 * len(overlap) / len(np.unique(mated_ids))
    if overlap_percent != 100.0:
        raise RuntimeError(
            f"Gallery/mated identity overlap is {overlap_percent:.2f}%, not 100%."
        )

    mated_distances = cdist(mated_features, templates, metric="euclidean")
    unmated_distances = cdist(unmated_features, templates, metric="euclidean")
    mated_nearest_indices = np.argmin(mated_distances, axis=1)
    unmated_min_distances = np.min(unmated_distances, axis=1)
    mated_min_distances = np.min(mated_distances, axis=1)
    predicted_mated_ids = gallery_identity_ids[mated_nearest_indices]
    mated_correct = predicted_mated_ids == mated_ids

    fpir_values, tpir_values = make_open_set_curve(
        mated_min_distances, mated_correct, unmated_min_distances
    )
    if len(fpir_values) < 2:
        raise ValueError("Open-set identification curve has insufficient FPIR values.")
    auc_value = float(
        np.sum(
            np.diff(fpir_values)
            * (tpir_values[:-1] + tpir_values[1:])
            / 2.0
        )
    )

    print("Matched-identity pilot evaluation (CPU feature extraction).")
    print(f"Results folder: {BASE}")
    print(f"Gallery images: {len(gallery_features):,}")
    print(f"Selected identities: {len(selected_ids):,}")
    print(f"Gallery templates: {len(gallery_identity_ids):,}")
    print(f"Mated probes: {len(mated_features):,}")
    print(f"Unmated probes: {len(unmated_features):,}")
    print(
        f"Gallery/mated identity overlap: {overlap_percent:.2f}% "
        f"({len(overlap):,}/{len(np.unique(mated_ids)):,})"
    )
    print(f"Mated probes correctly identified at rank 1: {np.mean(mated_correct):.4f}")
    for target_fpir in (0.1, 0.2, 0.3):
        tpir = float(np.interp(target_fpir, fpir_values, tpir_values))
        print(f"TPIR@FPIR={target_fpir:.1f}: {tpir:.4f}")
    print(f"Open-set identification AUC: {auc_value:.4f}")
    print(
        f"Loaded filename counts: gallery={len(gallery_names):,}, "
        f"mated={len(mated_names):,}, unmated={len(unmated_names):,}"
    )
    print(
        "Exploratory 100-identity pilot only; this is not the full official "
        "QMUL-SurvFace benchmark."
    )


def main() -> int:
    try:
        evaluate()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Pilot evaluation failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
