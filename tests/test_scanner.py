"""Tests for scanner.py"""

from __future__ import annotations

import os

import pytest

from imgdedup.scanner import is_raw, scan_directories


def test_single_file_path_is_yielded(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"x")
    results = list(scan_directories([str(img)]))
    assert len(results) == 1
    assert results[0][0] == str(img)


def test_unsupported_extension_is_skipped(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"x")
    (tmp_path / "photo.jpg").write_bytes(b"x")
    results = list(scan_directories([str(tmp_path)]))
    assert len(results) == 1
    assert results[0][0].endswith("photo.jpg")


def test_custom_extensions_narrows_scan(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    results = list(scan_directories([str(tmp_path)], extensions=frozenset({".jpg"})))
    assert len(results) == 1
    assert results[0][0].endswith("a.jpg")


def test_is_raw_uppercase():
    assert is_raw("photo.CR2") is True
    assert is_raw("photo.NEF") is True
    assert is_raw("photo.ARW") is True


def test_is_raw_false_for_raster():
    assert is_raw("photo.jpg") is False
    assert is_raw("photo.png") is False


def test_broken_symlink_is_skipped(tmp_path):
    link = tmp_path / "broken.jpg"
    link.symlink_to(tmp_path / "nonexistent.jpg")
    results = list(scan_directories([str(tmp_path)]))
    assert results == []
