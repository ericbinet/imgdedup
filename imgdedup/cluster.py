"""Union-find grouping of duplicate pairs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .compare import ScoredPair
from .hasher import ImageRecord


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        """Return {root: [members including root]}."""
        result: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return result


# ---------------------------------------------------------------------------
# Duplicate group
# ---------------------------------------------------------------------------


@dataclass
class DuplicateGroup:
    canonical: str
    members: list[str]  # excludes canonical
    pairs: list[ScoredPair]
    max_score: float
    min_score: float


def select_canonical(
    group: list[str],
    records: dict[str, ImageRecord],
    strategy: str = "largest",
) -> str:
    """Pick the 'original' from a group of duplicate paths."""
    def _key(path: str) -> tuple:
        rec = records.get(path)
        if rec is None:
            return (0, 0, 0)
        area = rec.width * rec.height
        size = rec.file_size
        mtime = rec.mtime
        if strategy == "largest":
            return (area, size, -mtime)
        elif strategy == "oldest":
            return (-mtime, area, size)
        elif strategy == "newest":
            return (mtime, area, size)
        elif strategy == "highest_res":
            return (area, size, -mtime)
        return (area, size, -mtime)

    return max(group, key=_key)


def build_groups(
    scored_pairs: list[ScoredPair],
    records: dict[str, ImageRecord],
    min_score: float = 0.80,
    canonical_strategy: str = "largest",
) -> list[DuplicateGroup]:
    """Group duplicate pairs using Union-Find, return sorted groups."""
    uf = UnionFind()
    relevant = [p for p in scored_pairs if p.final_score >= min_score]

    for pair in relevant:
        uf.union(pair.path_a, pair.path_b)

    # Map pairs to their group root for fast lookup
    pair_map: dict[str, list[ScoredPair]] = {}
    for pair in relevant:
        root = uf.find(pair.path_a)
        pair_map.setdefault(root, []).append(pair)

    groups: list[DuplicateGroup] = []
    for root, members in uf.groups().items():
        if len(members) < 2:
            continue
        canonical = select_canonical(members, records, canonical_strategy)
        non_canonical = [m for m in members if m != canonical]
        group_pairs = pair_map.get(root, [])
        scores = [p.final_score for p in group_pairs]
        groups.append(
            DuplicateGroup(
                canonical=canonical,
                members=non_canonical,
                pairs=group_pairs,
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
            )
        )

    # Sort: largest groups first, then by max score
    groups.sort(key=lambda g: (-(len(g.members) + 1), -g.max_score))
    return groups
