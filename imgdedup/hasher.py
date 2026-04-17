"""Perceptual hashing and SQLite cache."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator

import imagehash
import numpy as np
from PIL import Image

from .scanner import is_raw

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    path: str
    file_size: int
    mtime: float
    phash: int
    dhash: int
    whash: int
    width: int
    height: int
    is_raw: bool
    # 64-dim HSV color histogram, L2-normalized, stored as float32 bytes.
    # Optional so records predating this field stay loadable.
    color_hist: bytes | None = None


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_records (
    path      TEXT PRIMARY KEY,
    file_size INTEGER NOT NULL,
    mtime     REAL    NOT NULL,
    phash     INTEGER NOT NULL,
    dhash     INTEGER NOT NULL,
    whash     INTEGER NOT NULL,
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    is_raw    INTEGER NOT NULL DEFAULT 0,
    scanned_at REAL   NOT NULL,
    color_hist BLOB
);

CREATE TABLE IF NOT EXISTS tile_hashes (
    path    TEXT    NOT NULL,
    tile_idx INTEGER NOT NULL,
    tx      INTEGER NOT NULL,
    ty      INTEGER NOT NULL,
    tw      INTEGER NOT NULL,
    th      INTEGER NOT NULL,
    phash   INTEGER NOT NULL,
    dhash   INTEGER NOT NULL,
    whash   INTEGER NOT NULL,
    PRIMARY KEY (path, tile_idx),
    FOREIGN KEY (path) REFERENCES image_records(path)
);

CREATE TABLE IF NOT EXISTS candidate_pairs (
    path_a        TEXT NOT NULL,
    path_b        TEXT NOT NULL,
    min_hash_dist INTEGER NOT NULL,
    hash_agreement REAL NOT NULL,
    PRIMARY KEY (path_a, path_b)
);

CREATE TABLE IF NOT EXISTS scored_pairs (
    path_a           TEXT NOT NULL,
    path_b           TEXT NOT NULL,
    ssim             REAL,
    histogram_corr   REAL,
    normalized_mse   REAL,
    crop_score       REAL,
    crop_bbox        TEXT,
    final_score      REAL NOT NULL,
    modification_hints TEXT,
    PRIMARY KEY (path_a, path_b)
);

CREATE INDEX IF NOT EXISTS idx_records_phash ON image_records(phash);
CREATE INDEX IF NOT EXISTS idx_tile_path     ON tile_hashes(path);
CREATE INDEX IF NOT EXISTS idx_scored_final  ON scored_pairs(final_score DESC);
"""

# Minimum image area to generate tile hashes (saves work on tiny images)
TILE_MIN_AREA = 250_000   # ~500×500
# Grid sizes to use for tiling (multiple grids = more coverage)
TILE_GRIDS = ((2, 2), (3, 3), (4, 4))


class HashCache:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # image_records
    # ------------------------------------------------------------------

    def get(self, path: str, file_size: int, mtime: float) -> ImageRecord | None:
        row = self._conn.execute(
            "SELECT phash,dhash,whash,width,height,is_raw,color_hist "
            "FROM image_records WHERE path=? AND file_size=? AND mtime=?",
            (path, file_size, mtime),
        ).fetchone()
        if row is None:
            return None
        phash, dhash, whash, width, height, raw, color_hist = row
        return ImageRecord(
            path=path,
            file_size=file_size,
            mtime=mtime,
            phash=_from_signed64(phash),
            dhash=_from_signed64(dhash),
            whash=_from_signed64(whash),
            width=width,
            height=height,
            is_raw=bool(raw),
            color_hist=color_hist,
        )

    def put_many(self, records: list[ImageRecord]) -> None:
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO image_records "
            "(path,file_size,mtime,phash,dhash,whash,width,height,is_raw,scanned_at,color_hist) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r.path, r.file_size, r.mtime,
                    r.phash, r.dhash, r.whash,
                    r.width, r.height, int(r.is_raw), now,
                    r.color_hist,
                )
                for r in records
            ],
        )
        self._conn.commit()

    def get_all(self) -> list[ImageRecord]:
        rows = self._conn.execute(
            "SELECT path,file_size,mtime,phash,dhash,whash,width,height,is_raw,color_hist "
            "FROM image_records"
        ).fetchall()
        return [
            ImageRecord(
                path=r[0], file_size=r[1], mtime=r[2],
                phash=_from_signed64(r[3]),
                dhash=_from_signed64(r[4]),
                whash=_from_signed64(r[5]),
                width=r[6], height=r[7], is_raw=bool(r[8]),
                color_hist=r[9],
            )
            for r in rows
        ]

    def filter_uncached(
        self, file_iter: Iterator[tuple[str, int, float]]
    ) -> tuple[list[tuple[str, int, float]], int]:
        """Return (files_needing_hash, cached_count).
        A file needs re-hashing if the image_record is missing, stale, or
        one of its derived fields (tile hashes, color histogram) isn't
        present yet.
        """
        uncached: list[tuple[str, int, float]] = []
        cached = 0
        for path, size, mtime in file_iter:
            rec = self.get(path, size, mtime)
            if rec is None:
                uncached.append((path, size, mtime))
            elif rec.width * rec.height >= TILE_MIN_AREA and not self.has_tile_hashes(path):
                uncached.append((path, size, mtime))
            elif rec.color_hist is None:
                uncached.append((path, size, mtime))
            else:
                cached += 1
        return uncached, cached

    # ------------------------------------------------------------------
    # tile_hashes
    # ------------------------------------------------------------------

    def store_tile_hashes(self, path: str, tiles: list[tuple]) -> None:
        """Store tile hashes for one image. tiles: [(tile_idx,tx,ty,tw,th,ph,dh,wh)]"""
        self._conn.execute("DELETE FROM tile_hashes WHERE path=?", (path,))
        self._conn.executemany(
            "INSERT INTO tile_hashes (path,tile_idx,tx,ty,tw,th,phash,dhash,whash) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(path, *t) for t in tiles],
        )
        self._conn.commit()

    def get_all_tile_hashes(self) -> list[tuple]:
        """Return all tile rows as (path,tile_idx,tx,ty,tw,th,phash,dhash,whash)."""
        rows = self._conn.execute(
            "SELECT path,tile_idx,tx,ty,tw,th,phash,dhash,whash FROM tile_hashes"
        ).fetchall()
        return [
            (r[0], r[1], r[2], r[3], r[4], r[5],
             _from_signed64(r[6]), _from_signed64(r[7]), _from_signed64(r[8]))
            for r in rows
        ]

    def has_tile_hashes(self, path: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM tile_hashes WHERE path=? LIMIT 1", (path,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # candidate_pairs
    # ------------------------------------------------------------------

    def store_candidates(self, pairs: list) -> None:  # list[CandidatePair]
        self._conn.execute("DELETE FROM candidate_pairs")
        self._conn.executemany(
            "INSERT OR REPLACE INTO candidate_pairs "
            "(path_a,path_b,min_hash_dist,hash_agreement) VALUES (?,?,?,?)",
            [(p.path_a, p.path_b, p.min_hash_distance, p.hash_agreement) for p in pairs],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # scored_pairs
    # ------------------------------------------------------------------

    def store_scored_pairs(self, pairs: list) -> None:  # list[ScoredPair]
        import json
        self._conn.execute("DELETE FROM scored_pairs")
        self._conn.executemany(
            "INSERT OR REPLACE INTO scored_pairs "
            "(path_a,path_b,ssim,histogram_corr,normalized_mse,"
            "crop_score,crop_bbox,final_score,modification_hints) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    p.path_a, p.path_b,
                    p.ssim, p.histogram_corr, p.normalized_mse,
                    p.crop_score,
                    json.dumps(list(p.crop_bbox)) if p.crop_bbox else None,
                    p.final_score,
                    json.dumps(p.modification_hints),
                )
                for p in pairs
            ],
        )
        self._conn.commit()

    def get_scored_pairs(self) -> list:  # list[ScoredPair]
        import json
        from .compare import ScoredPair
        rows = self._conn.execute(
            "SELECT path_a,path_b,ssim,histogram_corr,normalized_mse,"
            "crop_score,crop_bbox,final_score,modification_hints "
            "FROM scored_pairs"
        ).fetchall()
        result = []
        for r in rows:
            bbox = tuple(json.loads(r[6])) if r[6] else None
            hints = json.loads(r[8]) if r[8] else []
            result.append(
                ScoredPair(
                    path_a=r[0], path_b=r[1],
                    ssim=r[2], histogram_corr=r[3], normalized_mse=r[4],
                    crop_score=r[5], crop_bbox=bbox,
                    final_score=r[7], modification_hints=hints,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _load_image(path: str, raw: bool) -> tuple[Image.Image, int, int]:
    """Return (PIL Image in RGB, width, height)."""
    if raw:
        try:
            import rawpy  # type: ignore
        except ImportError:
            raise ImportError(
                f"rawpy is required to process RAW files ({path}). "
                "Install it with: pip install rawpy"
            )
        with rawpy.imread(path) as rp:
            arr = rp.postprocess(half_size=True)
        img = Image.fromarray(arr)
    else:
        img = Image.open(path)
        img.load()
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGB")
    w, h = img.size
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, w, h


def _to_signed64(n: int) -> int:
    """Convert an unsigned 64-bit int to a signed 64-bit int for SQLite storage."""
    n = n & 0xFFFFFFFFFFFFFFFF
    if n >= 0x8000000000000000:
        n -= 0x10000000000000000
    return n


def _from_signed64(n: int) -> int:
    """Convert a signed 64-bit int back to an unsigned value for Hamming distance."""
    return n & 0xFFFFFFFFFFFFFFFF


def compute_hashes(img: Image.Image) -> tuple[int, int, int]:
    """Return (phash, dhash, whash) as signed 64-bit ints suitable for SQLite."""
    ph = _to_signed64(int(str(imagehash.phash(img)), 16))
    dh = _to_signed64(int(str(imagehash.dhash(img)), 16))
    wh = _to_signed64(int(str(imagehash.whash(img)), 16))
    return ph, dh, wh


def compute_color_histogram(img: Image.Image) -> bytes:
    """
    64-dim HSV histogram (4×4×4 bins), L2-normalized, as float32 bytes.
    Used as a cheap pre-filter for size-mismatched crop candidate search.
    """
    import cv2
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [4, 4, 4],
        [0, 180, 0, 256, 0, 256],
    ).flatten().astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm > 0:
        hist = hist / norm
    return hist.tobytes()


def compute_tile_hashes(
    img: Image.Image,
    grids: tuple[tuple[int, int], ...] = TILE_GRIDS,
) -> list[tuple]:
    """
    Compute perceptual hashes for tiles of img at multiple grid resolutions.
    Returns list of (tile_idx, tx, ty, tw, th, phash, dhash, whash).
    Only called for images with area >= TILE_MIN_AREA.
    """
    w, h = img.size
    tiles: list[tuple] = []
    tile_idx = 0
    seen_regions: set[tuple[int, int, int, int]] = set()

    for cols, rows in grids:
        tw = w // cols
        th = h // rows
        if tw < 64 or th < 64:
            continue
        for row in range(rows):
            for col in range(cols):
                tx = col * tw
                ty = row * th
                # Last tile in each row/col extends to edge
                actual_tw = w - tx if col == cols - 1 else tw
                actual_th = h - ty if row == rows - 1 else th
                region = (tx, ty, actual_tw, actual_th)
                if region in seen_regions:
                    tile_idx += 1
                    continue
                seen_regions.add(region)
                tile_img = img.crop((tx, ty, tx + actual_tw, ty + actual_th))
                ph, dh, wh = compute_hashes(tile_img)
                tiles.append((tile_idx, tx, ty, actual_tw, actual_th, ph, dh, wh))
                tile_idx += 1

    return tiles


@dataclass
class HashResult:
    record: ImageRecord
    tiles: list[tuple]  # (tile_idx, tx, ty, tw, th, phash, dhash, whash)


def hash_file(path: str, file_size: int, mtime: float) -> HashResult | Exception:
    """Compute ImageRecord + tile hashes for one file. Returns Exception on failure."""
    try:
        raw = is_raw(path)
        img, w, h = _load_image(path, raw)
        ph, dh, wh = compute_hashes(img)
        color_hist = compute_color_histogram(img)
        record = ImageRecord(
            path=path,
            file_size=file_size,
            mtime=mtime,
            phash=ph,
            dhash=dh,
            whash=wh,
            width=w,
            height=h,
            is_raw=raw,
            color_hist=color_hist,
        )
        tiles: list[tuple] = []
        if w * h >= TILE_MIN_AREA:
            tiles = compute_tile_hashes(img)
        return HashResult(record=record, tiles=tiles)
    except Exception as e:
        return e


def hash_files_parallel(
    files: list[tuple[str, int, float]],
    workers: int = 8,
    progress: bool = True,
) -> tuple[list[HashResult], list[tuple[str, Exception]]]:
    """Hash files using a thread pool. Returns (results, errors)."""
    from tqdm import tqdm

    results: list[HashResult] = []
    errors: list[tuple[str, Exception]] = []

    if not files:
        return results, errors

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(hash_file, p, s, m): p for p, s, m in files}
        it = tqdm(as_completed(futures), total=len(futures), desc="Hashing", unit="img", disable=not progress)
        for fut in it:
            path = futures[fut]
            result = fut.result()
            if isinstance(result, Exception):
                errors.append((path, result))
            else:
                results.append(result)

    return results, errors
