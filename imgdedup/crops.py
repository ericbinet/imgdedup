"""Crop detection: ORB+RANSAC homography with CLAHE normalization, template fallback."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .compare import (
    ScoredPair,
    compute_histogram_correlation,
    compute_normalized_mse,
    compute_ssim,
    sigmoid_sharpen,
)
from .hasher import ImageRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_SIDE = 1024  # downsample to this before feature extraction


def _pil_to_bgr(path: str, raw: bool, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Load image as uint8 BGR (OpenCV convention), downsampled if large."""
    if raw:
        try:
            import rawpy  # type: ignore
        except ImportError:
            raise ImportError("rawpy is required for RAW file support. Install: pip install rawpy")
        with rawpy.imread(path) as rp:
            arr = rp.postprocess(half_size=True)
        img = Image.fromarray(arr)
    else:
        img = Image.open(path).convert("RGB")
        img.load()

    # Downsample so max side ≤ max_side
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _equalize_luminance(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L channel of LAB colorspace to neutralise brightness differences."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Scale a (x,y,w,h) bbox from one image size to another."""
    sh, sw = from_shape[:2]
    th, tw = to_shape[:2]
    sx = tw / sw
    sy = th / sh
    x, y, w, h = bbox
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def _validate_homography(
    H: np.ndarray,
    large_shape: tuple[int, int],
    small_shape: tuple[int, int],
) -> bool:
    """
    Validate that H represents a plausible crop:
    - Scale factor in [0.15, 0.99]  (small is a sub-region of large)
    - No extreme perspective distortion
    - Mapped corners fall within large image bounds
    """
    # Estimate scale from top-left 2×2 sub-matrix
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if det <= 0:
        return False
    scale = float(det ** 0.5)
    if not (0.15 <= scale <= 0.99):
        return False

    # Map corners of small image through H and check they're inside large image
    sh, sw = small_shape[:2]
    lh, lw = large_shape[:2]
    corners = np.array([[0, 0, 1], [sw, 0, 1], [sw, sh, 1], [0, sh, 1]], dtype=np.float64)
    mapped = (H @ corners.T).T
    mapped /= mapped[:, 2:3]  # perspective divide

    xs, ys = mapped[:, 0], mapped[:, 1]
    if xs.min() < -lw * 0.15 or xs.max() > lw * 1.15:
        return False
    if ys.min() < -lh * 0.15 or ys.max() > lh * 1.15:
        return False

    return True


def _bbox_from_homography(
    H: np.ndarray,
    small_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Extract (x, y, w, h) bounding box of the small image mapped into the large image."""
    sh, sw = small_shape[:2]
    corners = np.array([[0, 0, 1], [sw, 0, 1], [sw, sh, 1], [0, sh, 1]], dtype=np.float64)
    mapped = (H @ corners.T).T
    mapped /= mapped[:, 2:3]

    xs, ys = mapped[:, 0], mapped[:, 1]
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    return x, y, w, h


# ---------------------------------------------------------------------------
# ORB-based detection
# ---------------------------------------------------------------------------


def detect_crop_orb(
    bgr_large: np.ndarray,
    bgr_small: np.ndarray,
    n_features: int = 2000,
    ratio_threshold: float = 0.75,
    min_inliers: int = 15,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """
    Returns (confidence, bbox_in_large_coords_or_None).
    bbox is in the coordinate space of bgr_large (the downsampled version).
    """
    eq_large = _equalize_luminance(bgr_large)
    eq_small = _equalize_luminance(bgr_small)

    gray_large = cv2.cvtColor(eq_large, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.cvtColor(eq_small, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features, scoreType=cv2.ORB_FAST_SCORE)
    kp_l, des_l = orb.detectAndCompute(gray_large, None)
    kp_s, des_s = orb.detectAndCompute(gray_small, None)

    if des_l is None or des_s is None or len(kp_l) < 8 or len(kp_s) < 8:
        return 0.0, None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_s, des_l, k=2)

    # Lowe ratio test
    good: list[cv2.DMatch] = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

    if len(good) < min_inliers:
        return float(len(good)) / min_inliers * 0.3, None

    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_l = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_s, pts_l, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return 0.1, None

    inliers = int(mask.sum())
    if inliers < min_inliers:
        return float(inliers) / min_inliers * 0.4, None

    if not _validate_homography(H, bgr_large.shape, bgr_small.shape):
        return 0.2, None

    # Confidence: sigmoid of inlier ratio
    inlier_ratio = inliers / len(good)
    confidence = sigmoid_sharpen(inlier_ratio, center=0.5, steepness=6.0)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    bbox = _bbox_from_homography(H, bgr_small.shape)
    return confidence, bbox


# ---------------------------------------------------------------------------
# Template matching fallback
# ---------------------------------------------------------------------------


def detect_crop_template(
    bgr_large: np.ndarray,
    bgr_small: np.ndarray,
    scales: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7),
) -> tuple[float, tuple[int, int, int, int] | None]:
    """
    Multi-scale normalized cross-correlation template matching.
    Returns (confidence, bbox_in_large_coords).
    """
    gray_large = cv2.cvtColor(bgr_large, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_small_orig = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    best_score = -1.0
    best_loc: tuple[int, int] | None = None
    best_tw, best_th = gray_small_orig.shape[1], gray_small_orig.shape[0]

    lh, lw = gray_large.shape

    for scale in scales:
        tw = int(gray_small_orig.shape[1] * scale)
        th = int(gray_small_orig.shape[0] * scale)
        if tw < 8 or th < 8 or tw > lw or th > lh:
            continue

        template = cv2.resize(gray_small_orig, (tw, th))
        result = cv2.matchTemplate(gray_large, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_tw, best_th = tw, th

    if best_loc is None or best_score < 0.4:
        return max(0.0, float(best_score)), None

    x, y = best_loc
    # Normalize score from [0.4, 1.0] → confidence
    confidence = sigmoid_sharpen(best_score, center=0.7, steepness=8.0)
    return float(np.clip(confidence, 0.0, 1.0)), (x, y, best_tw, best_th)


# ---------------------------------------------------------------------------
# Decision & integration
# ---------------------------------------------------------------------------


def should_run_crop_detection(
    rec_a: ImageRecord,
    rec_b: ImageRecord,
    scored: ScoredPair,
    size_ratio_threshold: float = 1.25,
    score_range: tuple[float, float] = (0.35, 0.85),
    via_tile: bool = False,
    via_orb: bool = False,
) -> bool:
    """Return True if we should attempt crop detection for this pair."""
    if via_tile or via_orb:
        # Tile or ORB search already confirmed this is a crop candidate;
        # crop detection materializes the bbox and refines the score.
        return True
    area_a = rec_a.width * rec_a.height
    area_b = rec_b.width * rec_b.height
    if area_a > 0 and area_b > 0:
        ratio = max(area_a, area_b) / min(area_a, area_b)
        if ratio >= size_ratio_threshold:
            return True
    lo, hi = score_range
    if lo <= scored.final_score <= hi:
        return True
    return False


def _compare_cropped_region(
    bgr_large: np.ndarray,
    bgr_small: np.ndarray,
    bbox_ds: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """
    Extract the bbox region from bgr_large (both in downsampled coords),
    resize it to match bgr_small, and return (ssim, hist_corr, normalized_mse)
    between the aligned region and the small image. This tells us whether
    the content INSIDE the bbox really matches the small image — critical for
    zoomed crops where full-image comparison is meaningless.
    """
    h_l, w_l = bgr_large.shape[:2]
    x, y, w, h = bbox_ds
    x = max(0, min(x, w_l - 1))
    y = max(0, min(y, h_l - 1))
    w = max(1, min(w, w_l - x))
    h = max(1, min(h, h_l - y))

    if w < 8 or h < 8:
        return 0.0, 0.0, 1.0

    cropped = bgr_large[y:y + h, x:x + w]
    h_s, w_s = bgr_small.shape[:2]
    aligned = cv2.resize(cropped, (w_s, h_s), interpolation=cv2.INTER_AREA)

    a_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    b_rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return (
        compute_ssim(a_rgb, b_rgb),
        compute_histogram_correlation(a_rgb, b_rgb),
        compute_normalized_mse(a_rgb, b_rgb),
    )


def run_crop_detection(
    rec_a: ImageRecord,
    rec_b: ImageRecord,
    scored: ScoredPair,
) -> ScoredPair:
    """
    Run ORB then template fallback. Updates scored in place with crop_score
    and crop_bbox. Bbox is in full-resolution coordinates of the larger image
    (either path_a or path_b, whichever is larger by area).

    When a bbox is found, we also compute similarity between the bbox region
    and the small image — this gives a meaningful final_score for zoomed crops
    where full-image SSIM is near zero.
    """
    if scored.crop_bbox is not None:
        # Already computed upstream; don't redo the work.
        return scored

    area_a = rec_a.width * rec_a.height
    area_b = rec_b.width * rec_b.height

    # Determine which is the "large" image (potential parent)
    if area_a >= area_b:
        rec_large, rec_small = rec_a, rec_b
    else:
        rec_large, rec_small = rec_b, rec_a

    try:
        bgr_large = _pil_to_bgr(rec_large.path, rec_large.is_raw)
        bgr_small = _pil_to_bgr(rec_small.path, rec_small.is_raw)
    except Exception:
        return scored

    # ORB first
    crop_score, bbox_ds = detect_crop_orb(bgr_large, bgr_small)

    # Template fallback if ORB didn't find a clear crop
    if bbox_ds is None or crop_score < 0.5:
        t_score, t_bbox = detect_crop_template(bgr_large, bgr_small)
        if t_score > crop_score:
            crop_score, bbox_ds = t_score, t_bbox

    if bbox_ds is not None:
        # Compute alignment quality: SSIM etc. on the bbox region resized to the
        # small image. For zoomed crops this is THE signal; for same-zoom crops
        # it reinforces the existing base score.
        ssim_crop, hist_crop, mse_crop = _compare_cropped_region(
            bgr_large, bgr_small, bbox_ds
        )
        alignment = 0.5 * ssim_crop + 0.25 * hist_crop + 0.25 * (1.0 - mse_crop)

        # Scale bbox to full-res coords of the larger image
        lh, lw = bgr_large.shape[:2]
        full_large_shape = (rec_large.height, rec_large.width)
        bbox_full = _scale_bbox(bbox_ds, (lh, lw), full_large_shape)

        hints = list(scored.modification_hints)
        if "crop" not in hints:
            hints.append("crop")

        # Combined final score: ORB confidence (geometric match) and
        # alignment (pixel-level similarity in the matched region) each
        # get equal weight. For zoomed crops, the alignment metric is noisy
        # because the crop region is small and heavily downsampled, so the
        # ORB confidence anchors the score. scored.final_score (full-image
        # comparison) is retained via max() below — for same-zoom crops it
        # may already be accurate enough.
        refined_raw = 0.5 * alignment + 0.5 * crop_score
        refined = sigmoid_sharpen(refined_raw, center=0.55, steepness=6.0)
        new_final = max(refined, scored.final_score)

        return ScoredPair(
            path_a=scored.path_a,
            path_b=scored.path_b,
            ssim=scored.ssim,
            histogram_corr=scored.histogram_corr,
            normalized_mse=scored.normalized_mse,
            crop_score=crop_score,
            crop_bbox=bbox_full,
            final_score=new_final,
            modification_hints=hints,
        )

    # No crop found — just attach the crop_score for reference
    return ScoredPair(
        path_a=scored.path_a,
        path_b=scored.path_b,
        ssim=scored.ssim,
        histogram_corr=scored.histogram_corr,
        normalized_mse=scored.normalized_mse,
        crop_score=crop_score if crop_score > 0 else None,
        crop_bbox=None,
        final_score=scored.final_score,
        modification_hints=scored.modification_hints,
    )
