"""Rich terminal output and JSON export."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

if TYPE_CHECKING:
    from .cluster import DuplicateGroup
    from .hasher import ImageRecord

console = Console()


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def _fmt_dims(rec: "ImageRecord") -> str:
    return f"{rec.width}×{rec.height}"


def _score_color(score: float) -> str:
    if score >= 0.95:
        return "bright_green"
    elif score >= 0.85:
        return "green"
    elif score >= 0.70:
        return "yellow"
    return "red"


def render_terminal(
    groups: list["DuplicateGroup"],
    records: dict[str, "ImageRecord"],
    min_score: float = 0.80,
    show_hints: bool = True,
) -> None:
    if not groups:
        console.print(Panel("[yellow]No duplicate groups found.[/yellow]", title="imgdedup"))
        return

    total_wasted = 0

    for i, group in enumerate(groups, 1):
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            title=f"[bold]Group {i}[/bold]  ({len(group.members) + 1} images)",
            title_style="bold white",
            expand=False,
        )
        table.add_column("Role", style="dim", width=10)
        table.add_column("Path", overflow="fold")
        table.add_column("Dims", justify="right", width=12)
        table.add_column("Size", justify="right", width=9)
        table.add_column("Score", justify="right", width=7)
        table.add_column("Hints", width=30)

        # Canonical row
        canon_rec = records.get(group.canonical)
        table.add_row(
            "[bold green]original[/bold green]",
            group.canonical,
            _fmt_dims(canon_rec) if canon_rec else "?",
            _fmt_bytes(canon_rec.file_size) if canon_rec else "?",
            "—",
            "",
        )

        # Build score lookup for display.
        # Prefer pairs that involve the canonical; fall back to any pair for
        # transitively-connected members that have no direct canonical pair.
        score_map: dict[str, tuple[float, list[str], tuple | None]] = {}
        for pair in group.pairs:
            if pair.path_a == group.canonical:
                score_map[pair.path_b] = (pair.final_score, pair.modification_hints, pair.crop_bbox)
            elif pair.path_b == group.canonical:
                score_map[pair.path_a] = (pair.final_score, pair.modification_hints, pair.crop_bbox)
        # Second pass: fill in members with no direct canonical pair.
        # Prefer pairs where the OTHER member is already scored (closer to canonical),
        # and among those prefer the pair with the highest score.
        remaining = [m for m in group.members if m not in score_map]
        if remaining:
            # Sort pairs: those touching an already-scored member first, then by score desc
            def _pair_priority(p: object) -> tuple:
                other_a = p.path_a in score_map  # type: ignore[union-attr]
                other_b = p.path_b in score_map  # type: ignore[union-attr]
                has_anchor = other_a or other_b
                return (-int(has_anchor), -p.final_score)  # type: ignore[union-attr]

            sorted_pairs = sorted(group.pairs, key=_pair_priority)
            for pair in sorted_pairs:
                for path in (pair.path_a, pair.path_b):
                    if path != group.canonical and path not in score_map:
                        score_map[path] = (pair.final_score, pair.modification_hints, pair.crop_bbox)

        for member in group.members:
            rec = records.get(member)
            score, hints, bbox = score_map.get(member, (0.0, [], None))
            total_wasted += rec.file_size if rec else 0

            hint_str = ""
            if show_hints and hints:
                hint_str = ", ".join(hints)
                if bbox:
                    x, y, w, h = bbox
                    hint_str += f"  [dim]bbox({x},{y},{w}×{h})[/dim]"

            color = _score_color(score)
            table.add_row(
                "[yellow]duplicate[/yellow]",
                member,
                _fmt_dims(rec) if rec else "?",
                _fmt_bytes(rec.file_size) if rec else "?",
                f"[{color}]{score:.2f}[/{color}]",
                hint_str,
            )

        console.print(table)

    console.print(
        Panel(
            f"[bold]{len(groups)}[/bold] duplicate groups · "
            f"[bold]{sum(len(g.members) for g in groups)}[/bold] duplicates · "
            f"~[bold]{_fmt_bytes(total_wasted)}[/bold] reclaimable",
            title="Summary",
            style="cyan",
        )
    )


def export_json(
    groups: list["DuplicateGroup"],
    records: dict[str, "ImageRecord"],
    output_path: str,
) -> None:
    def _rec_dict(path: str) -> dict:
        rec = records.get(path)
        if rec is None:
            return {"path": path}
        return {
            "path": path,
            "width": rec.width,
            "height": rec.height,
            "file_size": rec.file_size,
            "is_raw": rec.is_raw,
        }

    out: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_groups": len(groups),
        "total_duplicates": sum(len(g.members) for g in groups),
        "groups": [],
    }

    for group in groups:
        score_map: dict[str, tuple] = {}
        for pair in group.pairs:
            if pair.path_a == group.canonical:
                score_map[pair.path_b] = (
                    pair.final_score, pair.ssim, pair.histogram_corr,
                    pair.normalized_mse, pair.crop_score,
                    list(pair.crop_bbox) if pair.crop_bbox else None,
                    pair.modification_hints,
                )
            elif pair.path_b == group.canonical:
                score_map[pair.path_a] = (
                    pair.final_score, pair.ssim, pair.histogram_corr,
                    pair.normalized_mse, pair.crop_score,
                    list(pair.crop_bbox) if pair.crop_bbox else None,
                    pair.modification_hints,
                )
        # Fallback for transitive members: prefer pairs anchored to already-scored members
        def _pair_priority_json(p: object) -> tuple:
            has_anchor = p.path_a in score_map or p.path_b in score_map  # type: ignore[union-attr]
            return (-int(has_anchor), -p.final_score)  # type: ignore[union-attr]

        for pair in sorted(group.pairs, key=_pair_priority_json):
            for path in (pair.path_a, pair.path_b):
                if path != group.canonical and path not in score_map:
                    score_map[path] = (
                        pair.final_score, pair.ssim, pair.histogram_corr,
                        pair.normalized_mse, pair.crop_score,
                        list(pair.crop_bbox) if pair.crop_bbox else None,
                        pair.modification_hints,
                    )

        duplicates = []
        for member in group.members:
            info = score_map.get(member, (0.0, None, None, None, None, None, []))
            d = _rec_dict(member)
            d.update(
                {
                    "score": round(info[0], 4),
                    "ssim": round(info[1], 4) if info[1] is not None else None,
                    "histogram_corr": round(info[2], 4) if info[2] is not None else None,
                    "normalized_mse": round(info[3], 4) if info[3] is not None else None,
                    "crop_score": round(info[4], 4) if info[4] is not None else None,
                    "crop_bbox": info[5],
                    "hints": info[6],
                }
            )
            duplicates.append(d)

        out["groups"].append(
            {
                "canonical": _rec_dict(group.canonical),
                "duplicates": duplicates,
            }
        )

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    console.print(f"[green]JSON report written to[/green] {output_path}")


def render_summary_stats(
    records: dict[str, "ImageRecord"],
    groups: list["DuplicateGroup"],
    elapsed_seconds: float,
) -> None:
    total_images = len(records)
    total_groups = len(groups)
    total_dupes = sum(len(g.members) for g in groups)
    wasted = sum(
        records[m].file_size
        for g in groups
        for m in g.members
        if m in records
    )
    console.print(
        Panel(
            f"Scanned [bold]{total_images}[/bold] images in [bold]{elapsed_seconds:.1f}s[/bold]\n"
            f"Found [bold]{total_groups}[/bold] groups · [bold]{total_dupes}[/bold] duplicates · "
            f"~[bold]{_fmt_bytes(wasted)}[/bold] reclaimable",
            title="imgdedup",
            style="blue",
        )
    )
