"""Tests for reporter.py"""

from __future__ import annotations

import json

import pytest

from imgdedup.cluster import DuplicateGroup
from imgdedup.compare import ScoredPair
from imgdedup.hasher import ImageRecord
from imgdedup.reporter import _fmt_bytes, _score_color, export_json, render_terminal


def _make_record(path, width=100, height=100, file_size=1000):
    return ImageRecord(
        path=path, file_size=file_size, mtime=1.0,
        phash=0, dhash=0, whash=0,
        width=width, height=height, is_raw=False, color_hist=None,
    )


def _make_scored(path_a, path_b, score=0.9):
    return ScoredPair(
        path_a=path_a, path_b=path_b,
        ssim=score, histogram_corr=score, normalized_mse=1.0 - score,
        crop_score=None, crop_bbox=None,
        final_score=score, modification_hints=["reencoded"],
    )


def _make_group(canonical, members, score=0.9):
    pairs = [_make_scored(canonical, m, score) for m in members]
    return DuplicateGroup(
        canonical=canonical,
        members=members,
        pairs=pairs,
        max_score=score,
        min_score=score,
    )


# ---------------------------------------------------------------------------
# _fmt_bytes
# ---------------------------------------------------------------------------

def test_fmt_bytes_bytes():
    assert _fmt_bytes(512) == "512.0 B"


def test_fmt_bytes_kb_boundary():
    assert _fmt_bytes(1024) == "1.0 KB"


def test_fmt_bytes_megabytes():
    assert _fmt_bytes(2 * 1024 * 1024) == "2.0 MB"


def test_fmt_bytes_gigabytes():
    assert _fmt_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"


# ---------------------------------------------------------------------------
# _score_color
# ---------------------------------------------------------------------------

def test_score_color_bright_green():
    assert _score_color(0.95) == "bright_green"
    assert _score_color(1.0) == "bright_green"


def test_score_color_green():
    assert _score_color(0.85) == "green"
    assert _score_color(0.94) == "green"


def test_score_color_yellow():
    assert _score_color(0.70) == "yellow"
    assert _score_color(0.84) == "yellow"


def test_score_color_red():
    assert _score_color(0.69) == "red"
    assert _score_color(0.0) == "red"


# ---------------------------------------------------------------------------
# render_terminal
# ---------------------------------------------------------------------------

def test_render_terminal_empty_groups_no_crash():
    render_terminal([], {})  # should print "no duplicates" panel without raising


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

def test_export_json_valid_structure(tmp_path):
    canonical = "/original.jpg"
    member = "/duplicate.jpg"
    group = _make_group(canonical, [member])
    records = {
        canonical: _make_record(canonical),
        member: _make_record(member),
    }
    out = str(tmp_path / "out.json")
    export_json([group], records, out)

    with open(out) as f:
        data = json.load(f)

    assert "generated_at" in data
    assert data["total_groups"] == 1
    assert data["total_duplicates"] == 1
    assert data["groups"][0]["canonical"]["path"] == canonical
    dup = data["groups"][0]["duplicates"][0]
    assert dup["path"] == member
    assert "score" in dup
    assert "hints" in dup


def test_export_json_scores_rounded(tmp_path):
    canonical = "/original.jpg"
    member = "/duplicate.jpg"
    group = _make_group(canonical, [member], score=0.123456789)
    records = {canonical: _make_record(canonical), member: _make_record(member)}
    out = str(tmp_path / "out.json")
    export_json([group], records, out)
    with open(out) as f:
        data = json.load(f)
    score = data["groups"][0]["duplicates"][0]["score"]
    assert score == round(0.123456789, 4)


def test_export_json_missing_record_no_crash(tmp_path):
    canonical = "/original.jpg"
    member = "/missing.jpg"
    group = _make_group(canonical, [member])
    records = {canonical: _make_record(canonical)}  # member absent
    out = str(tmp_path / "out.json")
    export_json([group], records, out)
    with open(out) as f:
        data = json.load(f)
    assert data["total_groups"] == 1
