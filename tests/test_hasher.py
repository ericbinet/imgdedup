"""Tests for hasher.py"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imgdedup.hasher import (
    TILE_MIN_AREA,
    HashCache,
    ImageRecord,
    _from_signed64,
    _load_image,
    _to_signed64,
    compute_color_histogram,
    compute_hashes,
    compute_tile_hashes,
    hash_file,
)


def _make_image(width=100, height=100, color=(128, 64, 32)):
    return Image.new("RGB", (width, height), color)


def _make_record(path, width=100, height=100, file_size=1000, mtime=1.0, color_hist=b""):
    img = _make_image(width, height)
    ph, dh, wh = compute_hashes(img)
    return ImageRecord(
        path=path, file_size=file_size, mtime=mtime,
        phash=ph, dhash=dh, whash=wh,
        width=width, height=height, is_raw=False,
        color_hist=color_hist if color_hist != b"" else compute_color_histogram(img),
    )


# ---------------------------------------------------------------------------
# _to_signed64 / _from_signed64
# ---------------------------------------------------------------------------

def test_signed64_roundtrip():
    for n in [0, 1, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF]:
        assert _from_signed64(_to_signed64(n)) == n


# ---------------------------------------------------------------------------
# HashCache — cache hit / miss
# ---------------------------------------------------------------------------

def test_cache_hit(tmp_path):
    db = HashCache(str(tmp_path / "test.db"))
    rec = _make_record("/a.jpg")
    db.put_many([rec])
    result = db.get("/a.jpg", rec.file_size, rec.mtime)
    assert result is not None
    assert result.path == "/a.jpg"
    db.close()


def test_cache_miss_stale_mtime(tmp_path):
    db = HashCache(str(tmp_path / "test.db"))
    rec = _make_record("/a.jpg", mtime=1.0)
    db.put_many([rec])
    assert db.get("/a.jpg", rec.file_size, 2.0) is None
    db.close()


# ---------------------------------------------------------------------------
# filter_uncached
# ---------------------------------------------------------------------------

def test_filter_uncached_small_image_not_requeued(tmp_path):
    db = HashCache(str(tmp_path / "test.db"))
    rec = _make_record("/small.jpg", width=100, height=100)
    assert rec.width * rec.height < TILE_MIN_AREA
    db.put_many([rec])
    uncached, cached = db.filter_uncached(iter([("/small.jpg", rec.file_size, rec.mtime)]))
    assert cached == 1
    assert uncached == []
    db.close()


def test_filter_uncached_large_image_missing_tiles_requeued(tmp_path):
    db = HashCache(str(tmp_path / "test.db"))
    rec = _make_record("/large.jpg", width=1000, height=1000)
    assert rec.width * rec.height >= TILE_MIN_AREA
    db.put_many([rec])
    # No tile hashes stored → should be re-queued
    uncached, cached = db.filter_uncached(iter([("/large.jpg", rec.file_size, rec.mtime)]))
    assert len(uncached) == 1
    assert cached == 0
    db.close()


def test_filter_uncached_missing_color_hist_requeued(tmp_path):
    db = HashCache(str(tmp_path / "test.db"))
    img = _make_image()
    ph, dh, wh = compute_hashes(img)
    rec = ImageRecord(
        path="/a.jpg", file_size=1000, mtime=1.0,
        phash=ph, dhash=dh, whash=wh,
        width=100, height=100, is_raw=False, color_hist=None,
    )
    db.put_many([rec])
    uncached, cached = db.filter_uncached(iter([("/a.jpg", 1000, 1.0)]))
    assert len(uncached) == 1
    db.close()


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------

def test_hash_file_returns_exception_on_corrupt_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    result = hash_file(str(bad), 19, 0.0)
    assert isinstance(result, Exception)


# ---------------------------------------------------------------------------
# compute_tile_hashes
# ---------------------------------------------------------------------------

def test_compute_tile_hashes_skips_small_tiles():
    img = _make_image(40, 40)
    tiles = compute_tile_hashes(img)
    assert tiles == []


def test_compute_tile_hashes_deduplicates_regions():
    img = _make_image(512, 512)
    tiles = compute_tile_hashes(img)
    regions = [(t[1], t[2], t[3], t[4]) for t in tiles]  # tx, ty, tw, th
    assert len(regions) == len(set(regions))


# ---------------------------------------------------------------------------
# compute_color_histogram
# ---------------------------------------------------------------------------

def test_compute_color_histogram_solid_black_no_crash():
    # Solid black has all pixels in one bin; zero-norm guard prevents divide-by-zero
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    hist_bytes = compute_color_histogram(img)
    assert len(hist_bytes) == 64 * 4  # 64 float32s
    hist = np.frombuffer(hist_bytes, dtype=np.float32)
    assert not np.any(np.isnan(hist))


# ---------------------------------------------------------------------------
# _load_image — mode conversion
# ---------------------------------------------------------------------------

def test_load_image_rgba_converted_to_rgb(tmp_path):
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    path = str(tmp_path / "test.png")
    img.save(path)
    loaded, w, h = _load_image(path, raw=False)
    assert loaded.mode == "RGB"


def test_load_image_grayscale_converted_to_rgb(tmp_path):
    img = Image.new("L", (50, 50), 128)
    path = str(tmp_path / "gray.png")
    img.save(path)
    loaded, w, h = _load_image(path, raw=False)
    assert loaded.mode == "RGB"
