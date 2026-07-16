from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .config_loader import MazeWorldConfig
from .maze_geometry import WallBox


class MazePreviewRenderer:
    """Render a top-down preview of the generated maze walls."""

    def __init__(self, config: MazeWorldConfig) -> None:
        self.config = config

    def render(self, walls: Iterable[WallBox], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        figure_size = self.config.visualization.figure_size
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#ffffff")

        for wall in walls:
            lower_left_x = wall.center_x - 0.5 * wall.size_x
            lower_left_y = wall.center_y - 0.5 * wall.size_y
            ax.add_patch(
                Rectangle(
                    (lower_left_x, lower_left_y),
                    wall.size_x,
                    wall.size_y,
                    facecolor="#4b3f35",
                    edgecolor="#2d251f",
                    linewidth=0.7,
                    alpha=self.config.visualization.wall_alpha,
                )
            )

        margin = self.config.world.wall_thickness * 1.5
        ax.set_xlim(-margin, self.config.world.width + margin)
        ax.set_ylim(-margin, self.config.world.height + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Maze Preview ({self.config.maze.cols}x{self.config.maze.rows}, corridor={self.config.layout.corridor_width:.3f} m)"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linestyle="--", color="#d8d1c7", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(path, dpi=self.config.visualization.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

