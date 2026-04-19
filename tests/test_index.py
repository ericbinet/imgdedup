"""Tests for index.py"""

from __future__ import annotations

import numpy as np
import pytest

from imgdedup.hasher import ImageRecord
from imgdedup.index import HammingIndex, _canonical_pair, find_size_mismatch_candidates


def _make_record(path, phash=0, dhash=0, whash=0, width=100, height=100, color_hist=None):
    return ImageRecord(
        path=path, file_size=1000, mtime=1.0,
        phash=phash, dhash=dhash, whash=whash,
        width=width, height=height, is_raw=False,
        color_hist=color_hist,
    )


def _make_hist():
    h = np.ones(64, dtype=np.float32)
    h /= np.linalg.norm(h)
    return h.tobytes()


# ---------------------------------------------------------------------------
# HammingIndex — full-image search
# ---------------------------------------------------------------------------

def test_query_candidates_empty_records():
    idx = HammingIndex([])
    assert idx.query_candidates() == []


def test_query_candidates_no_self_matches():
    rec = _make_record("/a.jpg", phash=0, dhash=0, whash=0)
    idx = HammingIndex([rec])
    result = idx.query_candidates()
    assert all(c.path_a != c.path_b for c in result)


def test_query_candidates_pair_appears_once():
    rec_a = _make_record("/a.jpg", phash=0, dhash=0, whash=0)
    rec_b = _make_record("/b.jpg", phash=0, dhash=0, whash=0)
    idx = HammingIndex([rec_a, rec_b])
    result = idx.query_candidates()
    pairs = [(c.path_a, c.path_b) for c in result]
    assert len(pairs) == len(set(pairs))


def test_hash_agreement_all_three_hashes():
    rec_a = _make_record("/a.jpg", phash=0, dhash=0, whash=0)
    rec_b = _make_record("/b.jpg", phash=0, dhash=0, whash=0)
    idx = HammingIndex([rec_a, rec_b])
    result = idx.query_candidates()
    assert len(result) == 1
    assert result[0].hash_agreement == pytest.approx(1.0)


def test_tile_match_suppressed_when_full_image_match_exists():
    rec_a = _make_record("/a.jpg", phash=0, dhash=0, whash=0)
    rec_b = _make_record("/b.jpg", phash=0, dhash=0, whash=0)
    tile_rows = [("/a.jpg", 0, 0, 0, 50, 50, 0, 0, 0)]
    idx = HammingIndex([rec_a, rec_b], tile_rows=tile_rows)
    result = idx.query_candidates()
    # Pair found via full-image; tile match should not create a duplicate
    assert len(result) == 1
    assert result[0].via_tile is False


# ---------------------------------------------------------------------------
# find_size_mismatch_candidates
# ---------------------------------------------------------------------------

def test_find_size_mismatch_candidates_single_record():
    result = find_size_mismatch_candidates(
        [_make_record("/a.jpg")], existing_pairs=set(), progress=False
    )
    assert result == []


def test_find_size_mismatch_candidates_no_area_mismatch():
    # Both same size — no pair qualifies
    rec_a = _make_record("/a.jpg", width=100, height=100)
    rec_b = _make_record("/b.jpg", width=100, height=100)
    result = find_size_mismatch_candidates(
        [rec_a, rec_b], existing_pairs=set(),
        area_ratio_threshold=4.0, progress=False,
    )
    assert result == []


def test_find_size_mismatch_candidates_skips_existing_pairs():
    rec_small = _make_record("/small.jpg", width=100, height=100)
    rec_large = _make_record("/large.jpg", width=500, height=500)
    pair = _canonical_pair("/small.jpg", "/large.jpg")
    result = find_size_mismatch_candidates(
        [rec_small, rec_large],
        existing_pairs={pair},
        area_ratio_threshold=4.0, progress=False,
    )
    assert result == []
