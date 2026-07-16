from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D

from modules.world_loader import Obstacle


def _add_obstacle_patch(ax: plt.Axes, obstacle: Obstacle) -> None:
    """Draw one world obstacle on the provided axes."""
    if obstacle.kind == "box":
        rect = Rectangle(
            (-obstacle.hx, -obstacle.hy),
            2.0 * obstacle.hx,
            2.0 * obstacle.hy,
            facecolor="#7a7a7a",
            edgecolor="#505050",
            linewidth=0.8,
            alpha=0.85,
        )
        transform = Affine2D().rotate(obstacle.yaw).translate(obstacle.cx, obstacle.cy) + ax.transData
        rect.set_transform(transform)
        ax.add_patch(rect)
        return

    if obstacle.kind == "cylinder":
        circle = Circle(
            (obstacle.cx, obstacle.cy),
            radius=obstacle.r,
            facecolor="#7a7a7a",
            edgecolor="#505050",
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(circle)
        return

    raise ValueError(f"Unsupported obstacle kind: {obstacle.kind}")


def save_trial_plot(
    output_path: Path,
    obstacles: list[Obstacle],
    free_area_polygons: list[np.ndarray],
    start: tuple[float, float],
    goal: tuple[float, float],
    method_paths: dict[str, np.ndarray | None],
    title: str,
) -> None:
    """Save one trial plot with world obstacles, start, goal, and all method paths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    ax.set_facecolor("#fbfbfb")

    for obstacle in obstacles:
        _add_obstacle_patch(ax, obstacle)

    for polygon_xy in free_area_polygons:
        if polygon_xy is None or len(polygon_xy) < 3:
            continue
        patch = Polygon(
            polygon_xy,
            closed=True,
            fill=True,
            facecolor="#90ee90",
            edgecolor="#1f7a1f",
            linewidth=1.0,
            alpha=0.5,
            zorder=2,
        )
        ax.add_patch(patch)

    method_colors = {
        "astar_qp_base": "#0068b4",
        "astar_qp_wide_space": "#2da44e",
        "astar_qp_full": "#d97706",
        "bitstar_ppm": "#d62728",
    }
    method_labels = {
        "astar_qp_base": "A*QP (base)",
        "astar_qp_wide_space": "A*QP (wide-space)",
        "astar_qp_full": "A*QP (full)",
        "bitstar_ppm": "BIT*",
    }
    for method_name, path_xy in method_paths.items():
        if path_xy is None or len(path_xy) == 0:
            continue
        ax.plot(
            path_xy[:, 0],
            path_xy[:, 1],
            color=method_colors.get(method_name, "#303030"),
            linewidth=1.8,
            zorder=3,
            label=method_labels.get(method_name, method_name),
        )

    ax.scatter(start[0], start[1], marker="o", s=35, color="#1a7f37", zorder=4, label="Start")
    ax.scatter(goal[0], goal[1], marker="x", s=45, color="#d1242f", zorder=4, label="Goal")

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#c8c8c8")
    ax.legend(loc="best")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

