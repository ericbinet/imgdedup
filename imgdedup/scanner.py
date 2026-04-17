"""Walk directories and enumerate image files."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    pass

RASTER_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}
)

RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}
)

SUPPORTED_EXTENSIONS: frozenset[str] = RASTER_EXTENSIONS | RAW_EXTENSIONS


def scan_directories(
    paths: list[str],
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
) -> Iterator[tuple[str, int, float]]:
    """Yield (abspath, file_size, mtime) for every image file found under paths."""
    for root_path in paths:
        root_path = os.path.abspath(root_path)
        if os.path.isfile(root_path):
            ext = os.path.splitext(root_path)[1].lower()
            if ext in extensions:
                st = os.stat(root_path)
                yield root_path, st.st_size, st.st_mtime
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Sort for deterministic ordering
            dirnames.sort()
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in extensions:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                yield fpath, st.st_size, st.st_mtime


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS
