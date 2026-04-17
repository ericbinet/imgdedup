"""End-to-end crop detection tests on real sample images."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgdedup.compare import score_pair
from imgdedup.crops import run_crop_detection, should_run_crop_detection
from imgdedup.hasher import HashCache, hash_files_parallel
from imgdedup.index import HammingIndex, find_size_mismatch_candidates

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_IMGS = REPO_ROOT / "test_imgs"
BAT = TEST_IMGS / "Bat.jpeg"
ORIGINAL = TEST_IMGS / "PXL_20211007_002831945_Original.jpg"


pytestmark = pytest.mark.skipif(
    not BAT.exists() or not ORIGINAL.exists(),
    reason=f"Sample images not found in {TEST_IMGS}",
)


@pytest.fixture
def hashed_db(tmp_path):
    """Hash both sample images into a fresh per-test DB."""
    db_path = str(tmp_path / "crop_test.db")
    db = HashCache(db_path)

    files = []
    for p in (BAT, ORIGINAL):
        stat = p.stat()
        files.append((str(p), stat.st_size, stat.st_mtime))

    results, errors = hash_files_parallel(files, workers=1, progress=False)
    assert not errors, f"Hashing failed: {errors}"

    records = [r.record for r in results]
    db.put_many(records)
    for r in results:
        if r.tiles:
            db.store_tile_hashes(r.record.path, r.tiles)

    yield db, records
    db.close()


def test_bat_is_detected_as_crop_of_original(hashed_db):
    """
    Bat.jpeg is a zoomed landscape crop of the large bat drawing in the
    upper-left of the portrait Original.jpg. The pipeline should:

      1. Surface the (Bat, Original) pair as a candidate
      2. Run crop detection on it
      3. Detect a crop with a bounding box in the upper-left of Original
    """
    db, records = hashed_db
    records_by_path = {r.path: r for r in records}
    bat_path = str(BAT)
    orig_path = str(ORIGINAL)
    canonical_pair = tuple(sorted([bat_path, orig_path]))

    # ---- Step 1: candidate search must surface the pair --------------------
    # Stage 1: hash/tile-based search (same-zoom crops only)
    tile_rows = db.get_all_tile_hashes()
    index = HammingIndex(records, tile_rows=tile_rows)
    candidates = index.query_candidates()

    # Stage 2: ORB-based search for zoomed crops. The Bat is only ~12% of the
    # Original's area, far beyond what hash/tile search can catch, so stage 2
    # is required for this test case.
    existing = {(c.path_a, c.path_b) for c in candidates}
    orb_candidates = find_size_mismatch_candidates(
        records,
        existing_pairs=existing,
        area_ratio_threshold=4.0,
        progress=False,
    )
    candidates.extend(orb_candidates)

    found_pairs = {(c.path_a, c.path_b) for c in candidates}
    assert canonical_pair in found_pairs, (
        f"Bat/Original pair was not surfaced by candidate search.\n"
        f"  tile rows indexed: {len(tile_rows)}\n"
        f"  stage-1 candidates: {len(candidates) - len(orb_candidates)}\n"
        f"  stage-2 ORB candidates: {len(orb_candidates)}\n"
        f"  all pairs: {found_pairs}"
    )
    candidate = next(
        c for c in candidates if (c.path_a, c.path_b) == canonical_pair
    )

    # ---- Step 2: pairwise scoring ------------------------------------------
    scored = score_pair(
        records_by_path[candidate.path_a],
        records_by_path[candidate.path_b],
        candidate,
    )

    # ---- Step 3: crop detection should be triggered ------------------------
    assert should_run_crop_detection(
        records_by_path[candidate.path_a],
        records_by_path[candidate.path_b],
        scored,
        via_tile=candidate.via_tile,
        via_orb=candidate.via_orb,
    ), (
        f"Crop detection was not triggered for this pair. "
        f"via_tile={candidate.via_tile}, via_orb={candidate.via_orb}, "
        f"final_score={scored.final_score:.3f}"
    )

    result = run_crop_detection(
        records_by_path[candidate.path_a],
        records_by_path[candidate.path_b],
        scored,
    )

    # ---- Step 4: a crop should have been detected --------------------------
    assert result.crop_bbox is not None, (
        f"No crop bbox was computed. crop_score={result.crop_score}, "
        f"hints={result.modification_hints}"
    )
    assert result.crop_score is not None and result.crop_score >= 0.4, (
        f"Crop confidence too low: {result.crop_score:.3f}"
    )
    assert "crop" in result.modification_hints, (
        f"'crop' hint missing from {result.modification_hints}"
    )

    # ---- Step 5: bbox should be in the upper-left region of the Original ---
    # Original is 3024 wide × 4032 tall (portrait). The bat is in the upper
    # left, roughly x<1500, y<1200. Allow generous slack for RANSAC noise.
    x, y, w, h = result.crop_bbox
    assert x < 1800, (
        f"Crop x={x} suggests the match is in the right half of Original"
    )
    assert y < 2000, (
        f"Crop y={y} suggests the match is in the bottom half of Original"
    )
    assert w > 200 and h > 200, f"Crop bbox is suspiciously small: {w}×{h}"
