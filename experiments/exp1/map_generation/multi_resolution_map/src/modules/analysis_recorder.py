import csv
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

from .quadtree_node import QuadtreeNode
from .occupancy_classifier import CellState


@dataclass
class AnalysisRecord:
    count: int
    quadtree_area: float
    coverage_ratio: float


class ResolutionAnalysisRecorder:
    def __init__(self, true_free_area: float) -> None:
        # Store time-series records during quadtree subdivision.
        self._records: list[AnalysisRecord] = []
        self._true_free_area = true_free_area

    def record_from_leaves(self, leaves: list[QuadtreeNode]) -> None:
        # Recompute statistics from current leaves for robustness.
        quadtree_area = 0.0

        for leaf in leaves:
            if leaf.state == CellState.FREE:
                quadtree_area += leaf.rect.width * leaf.rect.height

        coverage_ratio = (
            quadtree_area / self._true_free_area if self._true_free_area > 0.0 else 0.0
        )

        self._records.append(
            AnalysisRecord(
                count=len(leaves),
                quadtree_area=quadtree_area,
                coverage_ratio=coverage_ratio,
            )
        )

    @property
    def records(self) -> list[AnalysisRecord]:
        return self._records

    def export_csv(self, path: Path) -> None:
        # Export analysis records with the requested header.
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num_sensors", "quadtree_area", "coverage_ratio"])
            for r in self._records:
                writer.writerow([r.count, r.quadtree_area, r.coverage_ratio])

    def export_plot(self, path: Path, figure_size_inch: float = 8.0, dpi: int = 200) -> None:
        # Plot coverage ratio against the current quadtree leaf count.
        xs = [r.count for r in self._records]
        ys = [r.coverage_ratio for r in self._records]

        fig, ax = plt.subplots(figsize=(figure_size_inch, figure_size_inch * 0.6), dpi=dpi)
        ax.plot(xs, ys)
        ax.set_xlabel("Number of leaf cells")
        ax.set_ylabel("Quadtree free-area coverage ratio")
        ax.grid(True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
