"""Unit tests for crops.py — end-to-end crop test lives in test_crop_detection.py."""

from __future__ import annotations

import numpy as np
import pytest

from imgdedup.compare import ScoredPair
from imgdedup.crops import (
    _compare_cropped_region,
    _scale_bbox,
    _validate_homography,
    detect_crop_orb,
    detect_crop_template,
    run_crop_detection,
    should_run_crop_detection,
)
from imgdedup.hasher import ImageRecord


def _make_record(path, width=100, height=100):
    return ImageRecord(
        path=path, file_size=1000, mtime=1.0,
        phash=0, dhash=0, whash=0,
        width=width, height=height, is_raw=False, color_hist=None,
    )


def _make_scored(path_a, path_b, score=0.5, crop_bbox=None):
    return ScoredPair(
        path_a=path_a, path_b=path_b,
        ssim=score, histogram_corr=score, normalized_mse=1.0 - score,
        crop_score=None, crop_bbox=crop_bbox,
        final_score=score, modification_hints=[],
    )


def _blank(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# should_run_crop_detection
# ---------------------------------------------------------------------------

def test_should_run_via_tile_short_circuits():
    rec = _make_record("/a.jpg")
    scored = _make_scored("/a.jpg", "/b.jpg", score=0.95)  # outside [0.35, 0.85]
    assert should_run_crop_detection(rec, rec, scored, via_tile=True) is True


def test_should_run_via_orb_short_circuits():
    rec = _make_record("/a.jpg")
    scored = _make_scored("/a.jpg", "/b.jpg", score=0.95)
    assert should_run_crop_detection(rec, rec, scored, via_orb=True) is True


def test_should_run_score_at_lower_boundary():
    rec = _make_record("/a.jpg")
    scored = _make_scored("/a.jpg", "/b.jpg", score=0.35)
    assert should_run_crop_detection(rec, rec, scored) is True


def test_should_run_score_at_upper_boundary():
    rec = _make_record("/a.jpg")
    scored = _make_scored("/a.jpg", "/b.jpg", score=0.85)
    assert should_run_crop_detection(rec, rec, scored) is True


def test_should_run_area_ratio_at_threshold():
    rec_large = _make_record("/large.jpg", width=125, height=100)  # area 12500
    rec_small = _make_record("/small.jpg", width=100, height=100)  # area 10000 → ratio 1.25
    scored = _make_scored("/large.jpg", "/small.jpg", score=0.95)
    assert should_run_crop_detection(rec_large, rec_small, scored, size_ratio_threshold=1.25) is True


def test_should_not_run_when_no_triggers():
    rec = _make_record("/a.jpg")
    scored = _make_scored("/a.jpg", "/b.jpg", score=0.95)  # above range, no size diff
    assert should_run_crop_detection(rec, rec, scored) is False


# ---------------------------------------------------------------------------
# detect_crop_orb
# ---------------------------------------------------------------------------

def test_detect_crop_orb_blank_image_no_keypoints():
    bgr = _blank(20, 20)
    score, bbox = detect_crop_orb(bgr, bgr)
    assert score == 0.0
    assert bbox is None


# ---------------------------------------------------------------------------
# detect_crop_template
# ---------------------------------------------------------------------------

def test_detect_crop_template_small_larger_than_large():
    large = _blank(50, 50)
    small = _blank(100, 100)  # bigger than large → all scales skipped
    score, bbox = detect_crop_template(large, small)
    assert bbox is None


def test_detect_crop_template_returns_float_score():
    rng = np.random.default_rng(42)
    large = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    small = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    score, _ = detect_crop_template(large, small)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# run_crop_detection
# ---------------------------------------------------------------------------

def test_run_crop_detection_skips_if_bbox_already_set():
    rec = _make_record("/a.jpg")
    scored = _make_scored("/a.jpg", "/b.jpg", crop_bbox=(0, 0, 100, 100))
    result = run_crop_detection(rec, rec, scored)
    assert result is scored  # early return — same object, no recomputation


# ---------------------------------------------------------------------------
# _validate_homography
# ---------------------------------------------------------------------------

def test_validate_homography_negative_det():
    H = np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_homography(H, (200, 200), (100, 100)) is False


def test_validate_homography_scale_equals_one():
    # Identity → scale = 1.0, which is > 0.99
    H = np.eye(3, dtype=np.float64)
    assert _validate_homography(H, (200, 200), (100, 100)) is False


def test_validate_homography_scale_too_small():
    H = np.diag([0.05, 0.05, 1.0])
    assert _validate_homography(H, (2000, 2000), (100, 100)) is False


def test_validate_homography_valid_scale():
    # Scale ≈ 0.5 — inside [0.15, 0.99] and corners inside large image
    H = np.diag([0.5, 0.5, 1.0])
    large = (1000, 1000)
    small = (200, 200)
    assert _validate_homography(H, large, small) is True


# ---------------------------------------------------------------------------
# _compare_cropped_region
# ---------------------------------------------------------------------------

def test_compare_cropped_region_tiny_bbox_returns_sentinel():
    bgr_large = _blank(100, 100)
    bgr_small = _blank(50, 50)
    ssim, hist, mse = _compare_cropped_region(bgr_large, bgr_small, (0, 0, 4, 4))  # w=4 < 8
    assert ssim == 0.0
    assert hist == 0.0
    assert mse == 1.0


# ---------------------------------------------------------------------------
# _scale_bbox
# ---------------------------------------------------------------------------

def test_scale_bbox_identity():
    bbox = (10, 20, 100, 200)
    assert _scale_bbox(bbox, (500, 500), (500, 500)) == bbox


def test_scale_bbox_double():
    bbox = (10, 20, 100, 200)
    result = _scale_bbox(bbox, (500, 500), (1000, 1000))
    assert result == (20, 40, 200, 400)
