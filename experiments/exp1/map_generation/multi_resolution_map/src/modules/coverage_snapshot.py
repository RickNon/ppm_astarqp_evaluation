import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .geometry2d import AARect
from .occupancy_classifier import CellState
from .quadtree_node import QuadtreeNode


@dataclass(frozen=True)
class CoverageSnapshot:
    target_ratio: float
    reached_ratio: float
    leaf_count: int
    free_leaf_count: int
    free_area: float
    leaves: tuple[QuadtreeNode, ...]


class CoverageThresholdSnapshotRecorder:
    def __init__(self, true_free_area: float, targets: Iterable[float]) -> None:
        if not math.isfinite(true_free_area) or true_free_area <= 0.0:
            raise ValueError("true_free_area must be a positive finite value")

        normalized_targets = sorted(set(float(target) for target in targets))
        if not normalized_targets:
            raise ValueError("At least one coverage target is required")
        if any(not math.isfinite(target) or target <= 0.0 or target > 1.0 for target in normalized_targets):
            raise ValueError("Coverage targets must be finite values in the interval (0, 1]")

        self._true_free_area = true_free_area
        self._targets = tuple(normalized_targets)
        self._snapshots: dict[float, CoverageSnapshot] = {}

    def record_from_leaves(self, leaves: Sequence[QuadtreeNode]) -> None:
        # Capture the first depth-wide frontier that reaches each requested target.
        free_leaves = [leaf for leaf in leaves if leaf.state == CellState.FREE]
        free_area = sum(leaf.rect.width * leaf.rect.height for leaf in free_leaves)
        reached_ratio = free_area / self._true_free_area

        for target in self._targets:
            if target in self._snapshots or reached_ratio < target:
                continue

            self._snapshots[target] = CoverageSnapshot(
                target_ratio=target,
                reached_ratio=reached_ratio,
                leaf_count=len(leaves),
                free_leaf_count=len(free_leaves),
                free_area=free_area,
                leaves=_clone_leaves(leaves),
            )

    @property
    def targets(self) -> tuple[float, ...]:
        return self._targets

    @property
    def snapshots(self) -> dict[float, CoverageSnapshot]:
        return dict(self._snapshots)

    @property
    def is_complete(self) -> bool:
        return len(self._snapshots) == len(self._targets)

    @property
    def missing_targets(self) -> tuple[float, ...]:
        return tuple(target for target in self._targets if target not in self._snapshots)


def _clone_leaves(leaves: Sequence[QuadtreeNode]) -> tuple[QuadtreeNode, ...]:
    # Freeze the frontier because later quadtree expansion mutates the original nodes.
    frozen: list[QuadtreeNode] = []
    for leaf in leaves:
        frozen.append(
            QuadtreeNode(
                rect=AARect(
                    xmin=leaf.rect.xmin,
                    ymin=leaf.rect.ymin,
                    xmax=leaf.rect.xmax,
                    ymax=leaf.rect.ymax,
                ),
                depth=leaf.depth,
                state=leaf.state,
                children=[],
            )
        )
    return tuple(frozen)
