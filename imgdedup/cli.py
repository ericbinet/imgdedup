"""Command-line interface for imgdedup."""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_DB = "imgdedup.db"


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    from .hasher import HashCache, hash_files_parallel
    from .scanner import scan_directories
    from rich.console import Console

    console = Console()
    t0 = time.time()

    db = HashCache(args.db)

    console.print(f"[cyan]Scanning[/cyan] {args.paths} ...")
    all_files = list(scan_directories(args.paths))
    console.print(f"  Found {len(all_files)} image files")

    uncached, cached_count = db.filter_uncached(iter(all_files))
    console.print(f"  {cached_count} already cached · {len(uncached)} to hash")

    if uncached:
        results, errors = hash_files_parallel(uncached, workers=args.workers, progress=True)
        records = [r.record for r in results]
        db.put_many(records)
        for r in results:
            if r.tiles:
                db.store_tile_hashes(r.record.path, r.tiles)
        if errors:
            console.print(f"[yellow]  {len(errors)} errors:[/yellow]")
            for path, err in errors[:10]:
                console.print(f"    [dim]{path}[/dim]: {err}")
            if len(errors) > 10:
                console.print(f"    ... and {len(errors) - 10} more")
        console.print(
            f"[green]Done.[/green] Hashed {len(records)} images in {time.time()-t0:.1f}s"
        )
    else:
        console.print("[green]All images already cached.[/green]")

    db.close()
    return 0


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


def cmd_find(args: argparse.Namespace) -> int:
    from .compare import score_pairs_batch
    from .crops import run_crop_detection, should_run_crop_detection
    from .hasher import HashCache
    from .index import HammingIndex, find_size_mismatch_candidates
    from rich.console import Console
    from tqdm import tqdm

    console = Console()
    t0 = time.time()

    db = HashCache(args.db)
    records_list = db.get_all()
    if not records_list:
        console.print("[yellow]No images in cache. Run 'imgdedup scan' first.[/yellow]")
        db.close()
        return 1

    records = {r.path: r for r in records_list}
    tile_rows = db.get_all_tile_hashes()
    console.print(
        f"[cyan]Building index[/cyan] over {len(records_list)} images "
        f"({len(tile_rows)} tile hashes) ..."
    )

    # --- Stage 1: hash-based candidate search (same-zoom crops included) ---
    idx = HammingIndex(records_list, tile_rows=tile_rows)
    candidates = idx.query_candidates(
        threshold_phash=args.threshold_phash,
        threshold_dhash=args.threshold_dhash,
        threshold_whash=args.threshold_whash,
    )
    console.print(f"  {len(candidates)} candidates from hash/tile search")

    # --- Stage 2: ORB-based zoomed-crop search (opt-in) ---
    if args.find_crops:
        existing = {(c.path_a, c.path_b) for c in candidates}
        console.print(
            f"[cyan]Zoomed-crop search[/cyan] (area ratio ≥ {args.crop_min_ratio:.1f}x, "
            f"≤ {args.crop_max_per_image} candidates per image) ..."
        )
        orb_candidates = find_size_mismatch_candidates(
            records_list,
            existing_pairs=existing,
            area_ratio_threshold=args.crop_min_ratio,
            max_per_small=args.crop_max_per_image,
            min_inliers=args.crop_min_inliers,
            progress=True,
        )
        console.print(f"  {len(orb_candidates)} additional candidates from ORB search")
        candidates.extend(orb_candidates)

    if not candidates:
        console.print(
            "[yellow]No candidates — try raising thresholds, or pass --find-crops "
            "for zoomed-crop detection.[/yellow]"
        )
        db.close()
        return 0

    db.store_candidates(candidates)

    console.print(f"[cyan]Scoring[/cyan] {len(candidates)} pairs ...")
    scored, errors = score_pairs_batch(records, candidates, workers=args.workers, progress=True)

    if errors:
        console.print(f"[yellow]{len(errors)} scoring errors[/yellow]")

    # Lookups for the crop detection decision (per-pair flags)
    via_tile_map = {(c.path_a, c.path_b): c.via_tile for c in candidates}
    via_orb_map = {(c.path_a, c.path_b): c.via_orb for c in candidates}

    if not args.no_crops:
        crop_candidates = [
            s for s in scored
            if should_run_crop_detection(
                records[s.path_a], records[s.path_b], s,
                via_tile=via_tile_map.get((s.path_a, s.path_b), False),
                via_orb=via_orb_map.get((s.path_a, s.path_b), False),
            )
        ]
        if crop_candidates:
            console.print(f"[cyan]Crop detection[/cyan] on {len(crop_candidates)} pairs ...")
            updated: dict[tuple[str, str], object] = {}
            for s in tqdm(crop_candidates, desc="Crops", unit="pair"):
                result = run_crop_detection(records[s.path_a], records[s.path_b], s)
                updated[(s.path_a, s.path_b)] = result
            scored = [updated.get((s.path_a, s.path_b), s) for s in scored]  # type: ignore[assignment]

    db.store_scored_pairs(scored)

    above = sum(1 for s in scored if s.final_score >= args.min_score)
    console.print(
        f"[green]Done.[/green] {len(scored)} pairs scored · "
        f"{above} above threshold {args.min_score:.2f} · "
        f"{time.time()-t0:.1f}s"
    )
    db.close()
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from .cluster import build_groups
    from .hasher import HashCache
    from .reporter import export_json, render_terminal

    db = HashCache(args.db)
    records_list = db.get_all()
    scored = db.get_scored_pairs()
    db.close()

    if not scored:
        from rich.console import Console
        Console().print("[yellow]No scored pairs found. Run 'imgdedup find' first.[/yellow]")
        return 1

    records = {r.path: r for r in records_list}
    groups = build_groups(scored, records, min_score=args.min_score, canonical_strategy=args.keep)

    render_terminal(groups, records, min_score=args.min_score)

    if args.json:
        export_json(groups, records, args.json)

    return 0


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def cmd_clean(args: argparse.Namespace) -> int:
    from .cluster import build_groups
    from .hasher import HashCache
    from .reporter import console

    db = HashCache(args.db)
    records_list = db.get_all()
    scored = db.get_scored_pairs()
    db.close()

    records = {r.path: r for r in records_list}
    groups = build_groups(scored, records, min_score=args.min_score, canonical_strategy=args.keep)

    if not groups:
        console.print("[yellow]No duplicate groups to clean.[/yellow]")
        return 0

    to_delete = [m for g in groups for m in g.members]
    total_size = sum(records[m].file_size for m in to_delete if m in records)

    from .reporter import _fmt_bytes
    console.print(
        f"[bold]{len(to_delete)}[/bold] files to remove "
        f"(~{_fmt_bytes(total_size)} freed)"
    )

    if args.dry_run:
        for path in to_delete:
            console.print(f"  [dim]would delete:[/dim] {path}")
        console.print("[yellow]Dry run — no files deleted.[/yellow]")
        return 0

    # Confirm
    answer = input(f"Delete {len(to_delete)} files? [y/N] ").strip().lower()
    if answer != "y":
        console.print("Aborted.")
        return 0

    import send2trash  # type: ignore
    deleted, failed = 0, 0
    for path in to_delete:
        try:
            send2trash.send2trash(path)
            deleted += 1
        except Exception as e:
            console.print(f"[red]Failed[/red] {path}: {e}")
            failed += 1

    console.print(f"[green]Deleted {deleted} files[/green]" + (f" · {failed} failed" if failed else ""))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imgdedup",
        description="Find near-duplicate images in a large collection.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, metavar="PATH",
                        help=f"SQLite cache/results database (default: {DEFAULT_DB})")

    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Hash images and populate the cache")
    p_scan.add_argument("paths", nargs="+", metavar="PATH",
                        help="Directories or files to scan")
    p_scan.add_argument("--workers", type=int, default=8, metavar="N",
                        help="Number of parallel hashing threads (default: 8)")

    # find
    p_find = sub.add_parser("find", help="Find candidate pairs and score them")
    p_find.add_argument("--workers", type=int, default=4, metavar="N",
                        help="Number of parallel scoring processes (default: 4)")
    p_find.add_argument("--threshold-phash", type=int, default=10, metavar="N",
                        help="pHash Hamming distance threshold (default: 10)")
    p_find.add_argument("--threshold-dhash", type=int, default=10, metavar="N",
                        help="dHash Hamming distance threshold (default: 10)")
    p_find.add_argument("--threshold-whash", type=int, default=12, metavar="N",
                        help="wHash Hamming distance threshold (default: 12)")
    p_find.add_argument("--min-score", type=float, default=0.80, metavar="F",
                        help="Minimum final score to report (default: 0.80)")
    p_find.add_argument("--no-crops", action="store_true",
                        help="Skip crop detection on candidate pairs (faster)")
    p_find.add_argument(
        "--find-crops", action="store_true",
        help="Enable stage-2 ORB-based search for zoomed crops "
             "(slower; finds images that are small portions of a larger image)",
    )
    p_find.add_argument(
        "--crop-min-ratio", type=float, default=4.0, metavar="F",
        help="Area ratio threshold for stage-2 ORB search (default: 4.0)",
    )
    p_find.add_argument(
        "--crop-max-per-image", type=int, default=20, metavar="N",
        help="Max ORB candidates per small image after color-histogram pre-filter (default: 20)",
    )
    p_find.add_argument(
        "--crop-min-inliers", type=int, default=12, metavar="N",
        help="Minimum RANSAC inliers for a confirmed ORB crop match (default: 12)",
    )

    # report
    p_report = sub.add_parser("report", help="Display duplicate groups")
    p_report.add_argument("--min-score", type=float, default=0.80, metavar="F",
                          help="Minimum score threshold (default: 0.80)")
    p_report.add_argument("--json", metavar="PATH", default=None,
                          help="Export results to JSON file")
    p_report.add_argument("--keep", default="largest",
                          choices=["largest", "oldest", "newest", "highest_res"],
                          help="Strategy for selecting canonical image (default: largest)")

    # clean
    p_clean = sub.add_parser("clean", help="Delete duplicates (send to trash)")
    p_clean.add_argument("--min-score", type=float, default=0.90, metavar="F",
                         help="Minimum score to consider for deletion (default: 0.90)")
    p_clean.add_argument("--keep", default="largest",
                         choices=["largest", "oldest", "newest", "highest_res"],
                         help="Which image to keep in each group (default: largest)")
    p_clean.add_argument("--dry-run", action="store_true", default=True,
                         help="Show what would be deleted without deleting (default: True)")
    p_clean.add_argument("--force", action="store_true",
                         help="Actually delete files (disables dry-run)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --force overrides --dry-run for clean
    if args.command == "clean" and args.force:
        args.dry_run = False

    dispatch = {
        "scan": cmd_scan,
        "find": cmd_find,
        "report": cmd_report,
        "clean": cmd_clean,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
