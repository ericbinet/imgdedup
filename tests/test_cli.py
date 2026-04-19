"""Tests for cli.py"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imgdedup.cli import build_parser, cmd_clean, cmd_find, cmd_report
from imgdedup.compare import ScoredPair
from imgdedup.hasher import HashCache, ImageRecord


def _make_record(path, file_size=1000, mtime=1.0, width=100, height=100):
    return ImageRecord(
        path=path, file_size=file_size, mtime=mtime,
        phash=0, dhash=0, whash=0,
        width=width, height=height, is_raw=False, color_hist=None,
    )


def _make_scored(path_a, path_b, score=0.99):
    return ScoredPair(
        path_a=path_a, path_b=path_b,
        ssim=score, histogram_corr=score, normalized_mse=1.0 - score,
        crop_score=None, crop_bbox=None,
        final_score=score, modification_hints=[],
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_clean_dry_run_defaults_true():
    args = build_parser().parse_args(["clean"])
    assert args.dry_run is True


def test_clean_force_sets_dry_run_false():
    args = build_parser().parse_args(["clean", "--force"])
    # main() applies this; simulate it:
    if args.force:
        args.dry_run = False
    assert args.dry_run is False


def test_find_no_crops_flag():
    args = build_parser().parse_args(["find", "--no-crops"])
    assert args.no_crops is True


def test_find_find_crops_flag():
    args = build_parser().parse_args(["find", "--find-crops"])
    assert args.find_crops is True


# ---------------------------------------------------------------------------
# cmd_find — empty cache
# ---------------------------------------------------------------------------

def test_cmd_find_empty_cache_exits_1(tmp_path):
    args = build_parser().parse_args(["--db", str(tmp_path / "empty.db"), "find"])
    assert cmd_find(args) == 1


# ---------------------------------------------------------------------------
# cmd_report — no scored pairs
# ---------------------------------------------------------------------------

def test_cmd_report_no_scored_pairs_exits_1(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = HashCache(db_path)
    db.close()
    args = build_parser().parse_args(["--db", db_path, "report"])
    assert cmd_report(args) == 1


# ---------------------------------------------------------------------------
# cmd_clean — dry-run never calls send2trash
# ---------------------------------------------------------------------------

def test_cmd_clean_dry_run_no_send2trash(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = HashCache(db_path)
    db.close()
    args = build_parser().parse_args(["--db", db_path, "clean"])
    assert args.dry_run is True
    with patch("send2trash.send2trash") as mock_trash:
        cmd_clean(args)
        mock_trash.assert_not_called()


# ---------------------------------------------------------------------------
# cmd_clean — non-"y" answer aborts without deleting
# ---------------------------------------------------------------------------

def test_cmd_clean_non_y_answer_aborts(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = HashCache(db_path)
    db.put_many([_make_record("/a.jpg", file_size=1000), _make_record("/b.jpg", file_size=500)])
    db.store_scored_pairs([_make_scored("/a.jpg", "/b.jpg", score=0.99)])
    db.close()

    args = build_parser().parse_args(["--db", db_path, "clean", "--force"])
    args.dry_run = False

    with patch("builtins.input", return_value="n"), \
         patch("send2trash.send2trash") as mock_trash:
        cmd_clean(args)
        mock_trash.assert_not_called()
