"""Plot obstacles, proximity points, and PPM polygons in 2D."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

from modules.load_world import Obstacle
from modules.prox_detector import SensorProx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle, Circle, Polygon


def _plot_proximity_base(
    obstacles: Iterable[Obstacle],
    sensor_prox: Iterable[SensorProx],
    padding: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8))

    obs_list = list(obstacles)
    xs = []
    ys = []
    for o in obs_list:
        if o.kind == "box":
            xs.extend([o.cx - o.hx, o.cx + o.hx])
            ys.extend([o.cy - o.hy, o.cy + o.hy])
        else:
            xs.extend([o.cx - o.r, o.cx + o.r])
            ys.extend([o.cy - o.r, o.cy + o.r])

    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding

    for o in obs_list:
        if o.kind == "box":
            rect = Rectangle((o.cx - o.hx, o.cy - o.hy), 2.0 * o.hx, 2.0 * o.hy,
                             facecolor="#3a3a3a", edgecolor="none", alpha=1.0, zorder=1)
            rect.set_transform(transforms.Affine2D().rotate_around(o.cx, o.cy, o.yaw) + ax.transData)
            ax.add_patch(rect)
        else:
            circ = Circle((o.cx, o.cy), o.r, facecolor="#3a3a3a", edgecolor="none", alpha=1.0, zorder=1)
            ax.add_patch(circ)

    for sp in sensor_prox:
        sensor_xyyaw = sp.sensor_xyyaw
        prox = sp.prox
        prox_list = list(prox)
        px, py, _ = sensor_xyyaw
        if prox_list:
            ax.scatter([h.qx for h in prox_list], [h.qy for h in prox_list],
                       c="#d62728", s=20, label="proximity points", zorder=4)
        ax.scatter([px], [py], edgecolor="#1f77b4", facecolor="#1f77b4", marker="o", s=30, label="sensor", zorder=5)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    return fig, ax


def _save_figure_multiple_formats(
    fig: plt.Figure,
    output_path: str,
    formats: list[str],
) -> None:
    """Save figure in multiple formats."""
    if not output_path:
        return
    
    base_path = os.path.splitext(output_path)[0]
    for fmt in formats:
        formatted_path = f"{base_path}.{fmt}"
        fig.savefig(formatted_path, bbox_inches="tight")


def plot_proximity(
    obstacles: Iterable[Obstacle],
    sensor_prox: Iterable[SensorProx],
    show: bool = True,
    output_path: str | None = None,
    padding: float = 1.0,
    formats: list[str] = ["png", "eps"],
) -> None:

    fig, _ = _plot_proximity_base(
        obstacles=obstacles,
        sensor_prox=sensor_prox,
        padding=padding,
    )

    if output_path:
        _save_figure_multiple_formats(fig, output_path, formats)
    if show:
        plt.show()
    plt.close(fig)


def plot_proximity_with_ppm(
    obstacles: Iterable[Obstacle],
    sensor_prox: Iterable[SensorProx],
    ppm_polygons: Iterable[Iterable[Tuple[float, float]]],
    show: bool = True,
    output_path: str | None = None,
    padding: float = 1.0,
    formats: list[str] = ["png", "eps"],
) -> None:
    fig, ax = _plot_proximity_base(
        obstacles=obstacles,
        sensor_prox=sensor_prox,
        padding=padding,
    )

    for poly in ppm_polygons:
        pts = list(poly)
        if len(pts) < 3:
            continue
        patch = Polygon(pts, closed=True, fill=True, edgecolor="#2ca02c", facecolor="lightgreen", linewidth=1.0, alpha=0.6, zorder=3)
        ax.add_patch(patch)

    if output_path:
        _save_figure_multiple_formats(fig, output_path, formats)
    if show:
        plt.show()
    plt.close(fig)

