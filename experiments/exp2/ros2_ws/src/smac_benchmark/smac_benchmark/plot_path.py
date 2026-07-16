#!/usr/bin/env python3
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml


def plot_path_on_map(
    map_yaml_path: str,
    waypoints_xy: List[Tuple[float, float]],
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    output_path: str,
) -> None:
    with open(map_yaml_path, "r", encoding="utf-8") as f:
        map_meta = yaml.safe_load(f)

    image_path = map_meta.get("image")
    if not image_path:
        raise ValueError(f"'image' is missing in map yaml: {map_yaml_path}")

    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

    resolution = float(map_meta.get("resolution", 0.0))
    origin = map_meta.get("origin", [0.0, 0.0, 0.0])
    if resolution <= 0.0:
        raise ValueError(f"Invalid map resolution: {resolution}")
    if not isinstance(origin, list) or len(origin) < 2:
        raise ValueError(f"Invalid map origin: {origin}")

    img = mpimg.imread(image_path)
    if img.ndim == 3:
        img = img[:, :, 0]

    height, width = img.shape
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    extent = [
        origin_x,
        origin_x + width * resolution,
        origin_y,
        origin_y + height * resolution,
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray", origin="upper", extent=extent)

    if waypoints_xy:
        xs = [p[0] for p in waypoints_xy]
        ys = [p[1] for p in waypoints_xy]
        ax.plot(xs, ys, color="red", linewidth=2.0, label="Path")

    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=80, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="cyan", s=80, marker="x")
    ax.text(start_xy[0], start_xy[1], " S", color="lime", fontsize=12, weight="bold")
    ax.text(goal_xy[0], goal_xy[1], " G", color="cyan", fontsize=12, weight="bold")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Planned Path on Map")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


