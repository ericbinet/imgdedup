"""Pairwise image similarity scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as _ssim

from .hasher import ImageRecord
from .index import CandidatePair

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ScoredPair:
    path_a: str
    path_b: str
    ssim: float
    histogram_corr: float
    normalized_mse: float
    crop_score: float | None
    crop_bbox: tuple[int, int, int, int] | None  # (x, y, w, h) in path_a coords
    final_score: float
    modification_hints: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

_COMPARE_SIZE = (512, 512)


def _load_pil(path: str, raw: bool) -> Image.Image:
    if raw:
        try:
            import rawpy  # type: ignore
        except ImportError:
            raise ImportError("rawpy is required for RAW file support. Install: pip install rawpy")
        with rawpy.imread(path) as rp:
            arr = rp.postprocess(half_size=True)
        img = Image.fromarray(arr)
    else:
        img = Image.open(path)
        img.load()
    return img.convert("RGB")


def _resize_to(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.array(img.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0


def load_pair_normalized(
    rec_a: ImageRecord,
    rec_b: ImageRecord,
    target_size: tuple[int, int] = _COMPARE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both images as float32 RGB arrays of target_size."""
    img_a = _load_pil(rec_a.path, rec_a.is_raw)
    img_b = _load_pil(rec_b.path, rec_b.is_raw)
    return _resize_to(img_a, target_size), _resize_to(img_b, target_size)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """SSIM on the Y (luminance) channel of YCbCr. Inputs are float32 [0,1] RGB."""
    def _to_y(img: np.ndarray) -> np.ndarray:
        # BT.601 coefficients
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    y_a = _to_y(img_a)
    y_b = _to_y(img_b)
    score, _ = _ssim(y_a, y_b, data_range=1.0, full=True)
    return float(np.clip(score, 0.0, 1.0))


def compute_histogram_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """OpenCV histogram correlation on RGB. Inputs are float32 [0,1]."""
    a8 = (img_a * 255).astype(np.uint8)
    b8 = (img_b * 255).astype(np.uint8)
    total = 0.0
    for ch in range(3):
        ha = cv2.calcHist([a8], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b8], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        total += float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return float(np.clip(total / 3.0, 0.0, 1.0))


def compute_normalized_mse(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """MSE normalized to [0,1]; lower is more similar."""
    mse = float(np.mean((img_a - img_b) ** 2))
    # Max possible MSE for [0,1] images is 1.0; clamp anyway.
    return float(np.clip(mse, 0.0, 1.0))


def detect_grayscale(img: np.ndarray) -> bool:
    """True if the image is effectively grayscale (R≈G≈B)."""
    diff_rg = np.std(img[..., 0] - img[..., 1])
    diff_rb = np.std(img[..., 0] - img[..., 2])
    return bool(diff_rg < 0.02 and diff_rb < 0.02)


# ---------------------------------------------------------------------------
# Sigmoid sharpening
# ---------------------------------------------------------------------------


def sigmoid_sharpen(x: float, center: float = 0.65, steepness: float = 8.0) -> float:
    """Push ambiguous scores toward 0 or 1."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def _detect_hints(
    rec_a: ImageRecord,
    rec_b: ImageRecord,
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> list[str]:
    hints: list[str] = []

    gray_a = detect_grayscale(img_a)
    gray_b = detect_grayscale(img_b)
    if gray_a != gray_b:
        hints.append("grayscale")
    elif gray_a and gray_b:
        hints.append("both_grayscale")

    # Size difference
    area_a = rec_a.width * rec_a.height
    area_b = rec_b.width * rec_b.height
    if area_a > 0 and area_b > 0:
        ratio = max(area_a, area_b) / min(area_a, area_b)
        if ratio >= 1.25:
            hints.append("size_difference")

    # Re-encoding: same pixel dims but different file extension
    import os
    ext_a = os.path.splitext(rec_a.path)[1].lower()
    ext_b = os.path.splitext(rec_b.path)[1].lower()
    if ext_a != ext_b:
        hints.append("reencoded")

    return hints


def _luminance(img: np.ndarray) -> np.ndarray:
    """BT.601 luminance channel, returned as float32 [0,1] single-channel."""
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return y[..., np.newaxis]


def compute_luminance_histogram_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Histogram correlation on luminance channel only."""
    a8 = (_luminance(img_a)[..., 0] * 255).astype(np.uint8)
    b8 = (_luminance(img_b)[..., 0] * 255).astype(np.uint8)
    ha = cv2.calcHist([a8], [0], None, [64], [0, 256])
    hb = cv2.calcHist([b8], [0], None, [64], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(np.clip(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL), 0.0, 1.0))


def score_pair(
    rec_a: ImageRecord,
    rec_b: ImageRecord,
    candidate: CandidatePair,
) -> ScoredPair:
    img_a, img_b = load_pair_normalized(rec_a, rec_b)

    gray_a = detect_grayscale(img_a)
    gray_b = detect_grayscale(img_b)
    one_is_gray = gray_a != gray_b

    ssim_val = compute_ssim(img_a, img_b)
    mse_val = compute_normalized_mse(img_a, img_b)

    if one_is_gray:
        # One image is grayscale — compare on luminance only to avoid
        # penalizing color histogram mismatch that's inherent to the conversion.
        hist_corr = compute_luminance_histogram_correlation(img_a, img_b)
        base = (
            0.50 * ssim_val           # upweight SSIM (most reliable for gray)
            + 0.20 * hist_corr
            + 0.20 * (1.0 - mse_val)
            + 0.10 * candidate.hash_agreement
        )
    else:
        hist_corr = compute_histogram_correlation(img_a, img_b)
        base = (
            0.35 * ssim_val
            + 0.25 * hist_corr
            + 0.25 * (1.0 - mse_val)
            + 0.15 * candidate.hash_agreement
        )

    final = sigmoid_sharpen(base)

    hints = _detect_hints(rec_a, rec_b, img_a, img_b)

    return ScoredPair(
        path_a=candidate.path_a,
        path_b=candidate.path_b,
        ssim=ssim_val,
        histogram_corr=hist_corr,
        normalized_mse=mse_val,
        crop_score=None,
        crop_bbox=None,
        final_score=final,
        modification_hints=hints,
    )


# ---------------------------------------------------------------------------
# Batch scoring (CPU-bound — ProcessPoolExecutor with spawn)
# ---------------------------------------------------------------------------


def _score_pair_worker(args: tuple) -> ScoredPair | tuple[str, str, str]:
    """Top-level function for multiprocessing."""
    rec_a, rec_b, candidate = args
    try:
        return score_pair(rec_a, rec_b, candidate)
    except Exception as e:
        return (candidate.path_a, candidate.path_b, str(e))


def score_pairs_batch(
    records: dict[str, ImageRecord],
    candidates: list[CandidatePair],
    workers: int = 4,
    progress: bool = True,
) -> tuple[list[ScoredPair], list[tuple[str, str, str]]]:
    """Score all candidate pairs. Returns (scored, errors)."""
    import multiprocessing as mp
    from tqdm import tqdm

    if not candidates:
        return [], []

    args = [
        (records[c.path_a], records[c.path_b], c)
        for c in candidates
        if c.path_a in records and c.path_b in records
    ]

    scored: list[ScoredPair] = []
    errors: list[tuple[str, str, str]] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        it = tqdm(
            pool.imap_unordered(_score_pair_worker, args),
            total=len(args),
            desc="Scoring pairs",
            unit="pair",
            disable=not progress,
        )
        for result in it:
            if isinstance(result, ScoredPair):
                scored.append(result)
            else:
                errors.append(result)

    return scored, errors
