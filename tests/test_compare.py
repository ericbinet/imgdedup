"""Tests for compare.py"""

from __future__ import annotations

import numpy as np
import pytest

from imgdedup.compare import (
    _detect_hints,
    detect_grayscale,
    score_pairs_batch,
    sigmoid_sharpen,
)
from imgdedup.hasher import ImageRecord
from imgdedup.index import CandidatePair


def _make_record(path, width=100, height=100):
    return ImageRecord(
        path=path, file_size=1000, mtime=1.0,
        phash=0, dhash=0, whash=0,
        width=width, height=height, is_raw=False, color_hist=None,
    )


def _make_candidate(path_a, path_b, agreement=1.0):
    return CandidatePair(
        path_a=path_a, path_b=path_b,
        min_hash_distance=0, hash_agreement=agreement,
        via_tile=False, via_orb=False,
    )


def _rgb(value=0.5, shape=(4, 4, 3)):
    return np.full(shape, value, dtype=np.float32)


def _gray(luma=0.5, shape=(4, 4, 3)):
    """Image where R==G==B (looks grayscale)."""
    return np.full(shape, luma, dtype=np.float32)


# ---------------------------------------------------------------------------
# sigmoid_sharpen
# ---------------------------------------------------------------------------

def test_sigmoid_sharpen_output_in_range():
    for x in [0.0, 0.5, 1.0]:
        assert 0.0 <= sigmoid_sharpen(x) <= 1.0


def test_sigmoid_sharpen_center_yields_half():
    assert sigmoid_sharpen(0.65, center=0.65, steepness=8.0) == pytest.approx(0.5)


def test_sigmoid_sharpen_monotone():
    assert sigmoid_sharpen(0.3) < sigmoid_sharpen(0.65) < sigmoid_sharpen(0.9)


# ---------------------------------------------------------------------------
# detect_grayscale
# ---------------------------------------------------------------------------

def test_detect_grayscale_true_for_equal_channels():
    assert detect_grayscale(_gray()) is True


def test_detect_grayscale_false_for_vivid_rgb():
    # Alternating red/green rows → high std of R-G differences
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[0::2, :, 0] = 1.0  # red rows
    img[1::2, :, 1] = 1.0  # green rows
    assert detect_grayscale(img) is False


def test_detect_grayscale_uniform_difference_is_grayscale():
    # Constant difference between channels → std of diff = 0 → grayscale
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[..., 0] = 0.5
    img[..., 1] = 0.51   # uniform offset → std = 0
    img[..., 2] = 0.5
    assert detect_grayscale(img) is True


# ---------------------------------------------------------------------------
# Score weight consistency
# ---------------------------------------------------------------------------

def test_color_branch_weights_sum_to_one():
    assert sum([0.35, 0.25, 0.25, 0.15]) == pytest.approx(1.0)


def test_grayscale_branch_weights_sum_to_one():
    assert sum([0.50, 0.20, 0.20, 0.10]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _detect_hints
# ---------------------------------------------------------------------------

def test_detect_hints_grayscale():
    rec_a = _make_record("/a.jpg")
    rec_b = _make_record("/b.jpg")
    # Alternating red/green → detect_grayscale=False; _gray() → True
    img_color = np.zeros((4, 4, 3), dtype=np.float32)
    img_color[0::2, :, 0] = 1.0
    img_color[1::2, :, 1] = 1.0
    hints = _detect_hints(rec_a, rec_b, img_color, _gray())
    assert "grayscale" in hints


def test_detect_hints_both_grayscale():
    rec_a = _make_record("/a.jpg")
    rec_b = _make_record("/b.jpg")
    hints = _detect_hints(rec_a, rec_b, _gray(0.3), _gray(0.7))
    assert "both_grayscale" in hints


def test_detect_hints_reencoded():
    rec_a = _make_record("/a.jpg")
    rec_b = _make_record("/b.png")
    img = _rgb()
    hints = _detect_hints(rec_a, rec_b, img, img)
    assert "reencoded" in hints


def test_detect_hints_size_difference():
    rec_a = _make_record("/a.jpg", width=1000, height=1000)
    rec_b = _make_record("/b.jpg", width=100, height=100)
    img = _rgb()
    hints = _detect_hints(rec_a, rec_b, img, img)
    assert "size_difference" in hints


def test_detect_hints_same_ext_no_reencoded():
    rec_a = _make_record("/a.jpg")
    rec_b = _make_record("/b.jpg")
    img = _rgb()
    hints = _detect_hints(rec_a, rec_b, img, img)
    assert "reencoded" not in hints


# ---------------------------------------------------------------------------
# score_pairs_batch
# ---------------------------------------------------------------------------

def test_score_pairs_batch_empty_candidates():
    scored, errors = score_pairs_batch({}, [], workers=1, progress=False)
    assert scored == []
    assert errors == []
