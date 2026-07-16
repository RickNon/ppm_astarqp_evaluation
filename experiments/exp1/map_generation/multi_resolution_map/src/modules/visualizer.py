from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Polygon, Circle as CirclePatch

from .geometry2d import AARect
from .obstacles import BoxObstacle2D, CylinderObstacle2D, Obstacle2D
from .quadtree_node import QuadtreeNode
from .occupancy_classifier import CellState
from .models import Model, Box


FREE_CELL_ALPHA = 0.75


class QuadtreeVisualizer:
    def __init__(
        self,
        outer_domain: AARect,
        figure_size_inch: float,
        dpi: int,
    ) -> None:
        self._fig, self._ax = plt.subplots(figsize=(figure_size_inch, figure_size_inch), dpi=dpi)
        self._outer_domain = outer_domain

    def draw_walls(self, models: list[Model], wall_color: str, line_width: float) -> None:
        # Draw wall bodies explicitly so the original wall region is included.
        for m in models:
            if m.type != "wall":
                continue
            if not isinstance(m.geometry, Box):
                continue

            xmin = m.pose.x - 0.5 * m.geometry.size_x
            ymin = m.pose.y - 0.5 * m.geometry.size_y

            patch = Rectangle(
                (xmin, ymin),
                m.geometry.size_x,
                m.geometry.size_y,
                facecolor=wall_color,
                edgecolor=wall_color,
                linewidth=line_width,
            )
            self._ax.add_patch(patch)

    def draw_leaves(
        self,
        leaves: list[QuadtreeNode],
        occupied_color: str,
        free_color: str,
        mixed_color: str,
        boundary_color: str,
        line_width: float,
    ) -> None:
        # Draw quadtree leaf cells with explicit state-dependent fill.
        free_facecolor = to_rgba(free_color, alpha=FREE_CELL_ALPHA)

        for node in leaves:
            rect = node.rect
            if node.state == CellState.OCCUPIED:
                facecolor = occupied_color
            elif node.state == CellState.MIXED:
                facecolor = mixed_color
            else:
                facecolor = free_facecolor

            patch = Rectangle(
                (rect.xmin, rect.ymin),
                rect.width,
                rect.height,
                facecolor=facecolor,
                edgecolor=boundary_color,
                linewidth=line_width,
            )
            self._ax.add_patch(patch)

    def draw_obstacles(
        self,
        obstacles: list[Obstacle2D],
        boundary_color: str,
        line_width: float,
    ) -> None:
        # Optional overlay for debugging obstacle extraction.
        for obs in obstacles:
            if isinstance(obs, BoxObstacle2D):
                pts = [(p.x, p.y) for p in obs.shape.world_corners()]
                patch = Polygon(
                    pts,
                    closed=True,
                    fill=False,
                    edgecolor=boundary_color,
                    linewidth=line_width,
                )
                self._ax.add_patch(patch)

            elif isinstance(obs, CylinderObstacle2D):
                patch = CirclePatch(
                    (obs.shape.center.x, obs.shape.center.y),
                    obs.shape.radius,
                    fill=False,
                    edgecolor=boundary_color,
                    linewidth=line_width,
                )
                self._ax.add_patch(patch)

    def draw_uniform_grid(
        self,
        grid: list[list[int]],
        domain: AARect,
        resolution: float,
        occupied_color: str,
        free_color: str,
        boundary_color: str,
        line_width: float,
    ) -> None:
        # Draw reconstructed dense grid cells.
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        free_facecolor = to_rgba(free_color, alpha=FREE_CELL_ALPHA)

        for iy in range(height):
            for ix in range(width):
                x = domain.xmin + ix * resolution
                y = domain.ymin + iy * resolution
                facecolor = occupied_color if grid[iy][ix] == 1 else free_facecolor

                patch = Rectangle(
                    (x, y),
                    resolution,
                    resolution,
                    facecolor=facecolor,
                    edgecolor=boundary_color,
                    linewidth=line_width,
                )
                self._ax.add_patch(patch)

    def finalize(self, show_axes: bool) -> None:
        self._ax.set_xlim(self._outer_domain.xmin, self._outer_domain.xmax)
        self._ax.set_ylim(self._outer_domain.ymin, self._outer_domain.ymax)
        self._ax.set_aspect("equal", adjustable="box")

        if show_axes:
            # Add metric axis labels for geometric verification.
            self._ax.set_xlabel("x [m]")
            self._ax.set_ylabel("y [m]")
        else:
            self._ax.axis("off")

    def save(self, path: Path) -> None:
        self._fig.savefig(path, bbox_inches="tight")
        plt.close(self._fig)
