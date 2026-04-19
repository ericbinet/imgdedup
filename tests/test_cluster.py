"""Tests for cluster.py"""

from __future__ import annotations

import pytest

from imgdedup.cluster import UnionFind, build_groups, select_canonical
from imgdedup.compare import ScoredPair
from imgdedup.hasher import ImageRecord


def _make_record(path, width=100, height=100, file_size=1000, mtime=1.0):
    return ImageRecord(
        path=path, file_size=file_size, mtime=mtime,
        phash=0, dhash=0, whash=0,
        width=width, height=height, is_raw=False, color_hist=None,
    )


def _make_scored(path_a, path_b, score=0.9):
    return ScoredPair(
        path_a=path_a, path_b=path_b,
        ssim=score, histogram_corr=score, normalized_mse=1.0 - score,
        crop_score=None, crop_bbox=None,
        final_score=score, modification_hints=[],
    )


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------

def test_union_find_path_compression():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    uf.union("c", "d")
    root = uf.find("d")
    # After find(), d's parent should point directly to root
    assert uf._parent["d"] == root


def test_union_find_same_root_is_noop():
    uf = UnionFind()
    uf.union("a", "b")
    root_before = uf.find("a")
    uf.union("a", "b")
    assert uf.find("a") == root_before


def test_union_find_groups_correct():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("c", "d")
    groups = uf.groups()
    members_flat = sorted(m for members in groups.values() for m in members)
    assert members_flat == ["a", "b", "c", "d"]
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# build_groups
# ---------------------------------------------------------------------------

def test_build_groups_all_below_min_score():
    pairs = [_make_scored("/a.jpg", "/b.jpg", score=0.5)]
    records = {p: _make_record(p) for p in ["/a.jpg", "/b.jpg"]}
    assert build_groups(pairs, records, min_score=0.80) == []


def test_build_groups_single_member_excluded():
    # Need at least two nodes in a cluster to form a group
    pairs = [_make_scored("/a.jpg", "/b.jpg", score=0.9)]
    records = {p: _make_record(p) for p in ["/a.jpg", "/b.jpg"]}
    groups = build_groups(pairs, records, min_score=0.80)
    assert len(groups) == 1
    # canonical + members = 2
    assert len(groups[0].members) + 1 == 2


def test_build_groups_sorted_larger_first():
    # Group of 3: a-b-c. Group of 2: x-y.
    pairs = [
        _make_scored("/a.jpg", "/b.jpg", score=0.9),
        _make_scored("/b.jpg", "/c.jpg", score=0.9),
        _make_scored("/x.jpg", "/y.jpg", score=0.95),
    ]
    records = {p: _make_record(p) for p in ["/a.jpg", "/b.jpg", "/c.jpg", "/x.jpg", "/y.jpg"]}
    groups = build_groups(pairs, records, min_score=0.80)
    sizes = [len(g.members) + 1 for g in groups]
    assert sizes == sorted(sizes, reverse=True)


def test_build_groups_max_min_score():
    pairs = [
        _make_scored("/a.jpg", "/b.jpg", score=0.91),
        _make_scored("/b.jpg", "/c.jpg", score=0.99),
    ]
    records = {p: _make_record(p) for p in ["/a.jpg", "/b.jpg", "/c.jpg"]}
    groups = build_groups(pairs, records, min_score=0.80)
    assert len(groups) == 1
    assert groups[0].max_score == pytest.approx(0.99)
    assert groups[0].min_score == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# select_canonical
# ---------------------------------------------------------------------------

def test_select_canonical_oldest():
    records = {
        "/old.jpg": _make_record("/old.jpg", mtime=1.0),
        "/new.jpg": _make_record("/new.jpg", mtime=100.0),
    }
    assert select_canonical(["/old.jpg", "/new.jpg"], records, strategy="oldest") == "/old.jpg"


def test_select_canonical_newest():
    records = {
        "/old.jpg": _make_record("/old.jpg", mtime=1.0),
        "/new.jpg": _make_record("/new.jpg", mtime=100.0),
    }
    assert select_canonical(["/old.jpg", "/new.jpg"], records, strategy="newest") == "/new.jpg"


def test_select_canonical_largest_file():
    records = {
        "/small.jpg": _make_record("/small.jpg", file_size=100),
        "/large.jpg": _make_record("/large.jpg", file_size=10000),
    }
    assert select_canonical(["/small.jpg", "/large.jpg"], records, strategy="largest") == "/large.jpg"


def test_select_canonical_highest_res():
    records = {
        "/lo.jpg": _make_record("/lo.jpg", width=100, height=100),
        "/hi.jpg": _make_record("/hi.jpg", width=1000, height=1000),
    }
    assert select_canonical(["/lo.jpg", "/hi.jpg"], records, strategy="highest_res") == "/hi.jpg"


def test_select_canonical_missing_record_no_crash():
    result = select_canonical(["/a.jpg", "/b.jpg"], records={}, strategy="largest")
    assert result in ("/a.jpg", "/b.jpg")
