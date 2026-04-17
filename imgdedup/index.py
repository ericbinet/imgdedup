"""BK-tree based Hamming distance index for candidate pair search."""

from __future__ import annotations

from dataclasses import dataclass

import pybktree  # type: ignore

from .hasher import ImageRecord

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CandidatePair:
    path_a: str  # always path_a < path_b lexicographically
    path_b: str
    min_hash_distance: int
    hash_agreement: float  # fraction of the 3 hash types within threshold
    via_tile: bool = False  # True if found via tile hash match (crop candidate)
    via_orb: bool = False   # True if found via ORB feature search (zoomed crop)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

_HASH_NAMES = ("phash", "dhash", "whash")
_DEFAULT_THRESHOLDS = {"phash": 10, "dhash": 10, "whash": 12}


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _dist_fn(a: tuple[int, str], b: tuple[int, str]) -> int:
    return _hamming(a[0], b[0])


class HammingIndex:
    """
    Three BK-trees over full-image hashes + three BK-trees over tile hashes.
    Tile trees let small images match sub-regions of large images (crop detection).
    """

    def __init__(
        self,
        records: list[ImageRecord],
        tile_rows: list[tuple] | None = None,
    ) -> None:
        """
        records   — all ImageRecord objects
        tile_rows — rows from HashCache.get_all_tile_hashes():
                    (path, tile_idx, tx, ty, tw, th, phash, dhash, whash)
        """
        self._records = records
        self._by_path: dict[str, ImageRecord] = {r.path: r for r in records}

        # Full-image BK-trees
        self._trees: dict[str, pybktree.BKTree] = {}
        for name in _HASH_NAMES:
            items = [(getattr(r, name), r.path) for r in records]
            self._trees[name] = pybktree.BKTree(_dist_fn, items)

        # Tile BK-trees — each entry: (hash_value, parent_path)
        # (tile_idx / bbox stored separately for lookup)
        self._tile_trees: dict[str, pybktree.BKTree] = {}
        self._tile_meta: dict[tuple[str, int], tuple[int, int, int, int]] = {}

        if tile_rows:
            tile_items: dict[str, list[tuple[int, str]]] = {n: [] for n in _HASH_NAMES}
            for (path, tile_idx, tx, ty, tw, th, ph, dh, wh) in tile_rows:
                tile_items["phash"].append((ph, path))
                tile_items["dhash"].append((dh, path))
                tile_items["whash"].append((wh, path))
                self._tile_meta[(path, tile_idx)] = (tx, ty, tw, th)

            for name, items in tile_items.items():
                if items:
                    self._tile_trees[name] = pybktree.BKTree(_dist_fn, items)

    def query_candidates(
        self,
        threshold_phash: int = _DEFAULT_THRESHOLDS["phash"],
        threshold_dhash: int = _DEFAULT_THRESHOLDS["dhash"],
        threshold_whash: int = _DEFAULT_THRESHOLDS["whash"],
    ) -> list[CandidatePair]:
        thresholds = {
            "phash": threshold_phash,
            "dhash": threshold_dhash,
            "whash": threshold_whash,
        }

        # --- Full-image vs full-image candidates ---
        raw_hits: dict[tuple[str, str], dict[str, int]] = {}

        for name, tree in self._trees.items():
            thresh = thresholds[name]
            for rec in self._records:
                query_val = (getattr(rec, name), rec.path)
                for dist, (_, matched_path) in tree.find(query_val, thresh):
                    if matched_path == rec.path:
                        continue
                    pair = _canonical_pair(rec.path, matched_path)
                    raw_hits.setdefault(pair, {})[name] = min(
                        raw_hits.get(pair, {}).get(name, dist), dist
                    )

        full_pairs: dict[tuple[str, str], CandidatePair] = {}
        for (path_a, path_b), dists_by_type in raw_hits.items():
            min_dist = min(dists_by_type.values())
            agreement = sum(
                1 for n in _HASH_NAMES
                if dists_by_type.get(n, 999) <= thresholds[n]
            ) / len(_HASH_NAMES)
            full_pairs[(path_a, path_b)] = CandidatePair(
                path_a=path_a,
                path_b=path_b,
                min_hash_distance=min_dist,
                hash_agreement=agreement,
                via_tile=False,
            )

        # --- Full-image vs tile candidates (crop search) ---
        tile_pairs: dict[tuple[str, str], CandidatePair] = {}

        if self._tile_trees:
            for name, tile_tree in self._tile_trees.items():
                thresh = thresholds[name]
                for rec in self._records:
                    query_val = (getattr(rec, name), rec.path)
                    for dist, (_, parent_path) in tile_tree.find(query_val, thresh):
                        if parent_path == rec.path:
                            continue
                        pair = _canonical_pair(rec.path, parent_path)
                        if pair in full_pairs:
                            continue  # already found via full-image match
                        existing = tile_pairs.get(pair)
                        if existing is None or dist < existing.min_hash_distance:
                            tile_pairs[pair] = CandidatePair(
                                path_a=pair[0],
                                path_b=pair[1],
                                min_hash_distance=dist,
                                hash_agreement=1.0 / len(_HASH_NAMES),
                                via_tile=True,
                            )

        result = list(full_pairs.values()) + list(tile_pairs.values())
        result.sort(key=lambda p: (p.via_tile, p.min_hash_distance))
        return result


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------------------
# Stage 2: ORB-based size-mismatch candidate search
# ---------------------------------------------------------------------------


def find_size_mismatch_candidates(
    records: list[ImageRecord],
    existing_pairs: set[tuple[str, str]],
    *,
    area_ratio_threshold: float = 4.0,
    max_per_small: int = 20,
    min_confidence: float = 0.5,
    min_inliers: int = 12,
    progress: bool = True,
) -> list[CandidatePair]:
    """
    Find crop candidates among size-mismatched image pairs using ORB feature
    matching. Intended for "zoomed" crops (a small image is a sub-region of
    a much larger one) where perceptual hashing can't surface the relationship.

    Uses a color-histogram pre-filter (when available on records) to cap the
    number of ORB comparisons per small image at `max_per_small`.
    """
    import os
    import cv2  # noqa: F401 (ensures cv2 is loadable)
    import numpy as np
    from tqdm import tqdm

    from .crops import _pil_to_bgr, detect_crop_orb

    if len(records) < 2:
        return []

    # 1. Sort by area so iteration gives (small, large) naturally
    recs_sorted = sorted(records, key=lambda r: r.width * r.height)
    areas = np.array([r.width * r.height for r in recs_sorted], dtype=np.int64)

    # 2. Build histogram matrix once if all records have color_hist
    hist_matrix: np.ndarray | None = None
    if all(r.color_hist is not None for r in recs_sorted):
        try:
            hist_matrix = np.stack([
                np.frombuffer(r.color_hist, dtype=np.float32) for r in recs_sorted
            ])
            # Rows are already L2-normalized when written.
            if hist_matrix.shape[1] == 0:
                hist_matrix = None
        except Exception:
            hist_matrix = None

    # 3. For each "small" image, find candidate "large" images
    pairs_to_check: list[tuple[int, int]] = []  # indices into recs_sorted
    for si in range(len(recs_sorted)):
        small_area = int(areas[si])
        if small_area <= 0:
            continue
        # Find first index where large_area >= area_ratio_threshold * small_area
        min_large_area = small_area * area_ratio_threshold
        # linear scan is fine: small always comes before large after sort
        large_indices: list[int] = []
        for li in range(si + 1, len(recs_sorted)):
            if int(areas[li]) >= min_large_area:
                large_indices.append(li)
        if not large_indices:
            continue

        # Filter out pairs already in existing_pairs
        def _not_existing(li: int) -> bool:
            pair = _canonical_pair(recs_sorted[si].path, recs_sorted[li].path)
            return pair not in existing_pairs

        large_indices = [li for li in large_indices if _not_existing(li)]
        if not large_indices:
            continue

        # Histogram pre-filter: keep the top max_per_small most similar
        if hist_matrix is not None and len(large_indices) > max_per_small:
            small_vec = hist_matrix[si]
            sims = hist_matrix[large_indices] @ small_vec  # cosine on L2-normed vectors
            keep = np.argsort(-sims)[:max_per_small]
            large_indices = [large_indices[k] for k in keep]

        for li in large_indices:
            pairs_to_check.append((si, li))

    if not pairs_to_check:
        return []

    # 4. Load images lazily with a per-path cache (ORB descriptors are cheap to recompute)
    bgr_cache: dict[str, np.ndarray] = {}

    def _get_bgr(rec: ImageRecord) -> np.ndarray:
        if rec.path not in bgr_cache:
            bgr_cache[rec.path] = _pil_to_bgr(rec.path, rec.is_raw)
        return bgr_cache[rec.path]

    # 5. Run ORB on each pair; keep only confirmed matches
    candidates: list[CandidatePair] = []
    it = tqdm(
        pairs_to_check,
        desc="ORB crop search",
        unit="pair",
        disable=not progress,
    )
    for si, li in it:
        small_rec = recs_sorted[si]
        large_rec = recs_sorted[li]
        try:
            bgr_small = _get_bgr(small_rec)
            bgr_large = _get_bgr(large_rec)
        except Exception:
            continue

        try:
            score, bbox = detect_crop_orb(
                bgr_large,
                bgr_small,
                n_features=2000,
                min_inliers=min_inliers,
            )
        except Exception:
            continue

        if bbox is None or score < min_confidence:
            continue

        pair = _canonical_pair(small_rec.path, large_rec.path)
        candidates.append(
            CandidatePair(
                path_a=pair[0],
                path_b=pair[1],
                min_hash_distance=64,   # sentinel: hash-based search couldn't find it
                hash_agreement=0.0,
                via_tile=False,
                via_orb=True,
            )
        )

    return candidates
