import csv
import math
from pathlib import Path
from typing import Sequence

from .geometry2d import AARect
from .occupancy_classifier import CellState
from .quadtree_node import QuadtreeNode


def reconstruct_uniform_grid_from_leaves(
    leaves: Sequence[QuadtreeNode],
    domain: AARect,
    resolution: float,
    treat_mixed_as_occupied: bool = False,
) -> tuple[list[list[int]], int, int]:
    # Reconstruct a dense occupancy grid from leaf cells.
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be a positive finite value")

    width = int(round(domain.width / resolution))
    height = int(round(domain.height / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("The domain must contain at least one grid cell")

    grid = [[0 for _ in range(width)] for _ in range(height)]

    for leaf in leaves:
        value = int(
            leaf.state == CellState.OCCUPIED
            or (treat_mixed_as_occupied and leaf.state == CellState.MIXED)
        )

        ix0 = max(0, int(round((leaf.rect.xmin - domain.xmin) / resolution)))
        ix1 = min(width, int(round((leaf.rect.xmax - domain.xmin) / resolution)))
        iy0 = max(0, int(round((leaf.rect.ymin - domain.ymin) / resolution)))
        iy1 = min(height, int(round((leaf.rect.ymax - domain.ymin) / resolution)))

        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                grid[iy][ix] = value

    return grid, width, height


def export_uniform_grid_csv(
    grid: list[list[int]],
    csv_path: Path,
) -> None:
    # Export the reconstructed dense occupancy grid.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in grid:
            writer.writerow(row)
