from pathlib import Path
from typing import Sequence

import yaml

from .geometry2d import AARect
from .models import Model
from .obstacles import model_to_obstacle
from .quadtree_node import QuadtreeNode
from .uniform_grid import reconstruct_uniform_grid_from_leaves


def export_nav2_map_from_leaves(
    leaves: Sequence[QuadtreeNode],
    inner_domain: AARect,
    outer_domain: AARect,
    models: Sequence[Model],
    resolution: float,
    output_stem: Path,
    treat_mixed_as_occupied: bool = True,
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
    negate: int = 0,
    mode: str = "trinary",
) -> tuple[Path, Path]:
    # Convert a quadtree frontier into the wall-inclusive map consumed by Nav2.
    inner_grid, _, _ = reconstruct_uniform_grid_from_leaves(
        leaves=leaves,
        domain=inner_domain,
        resolution=resolution,
        treat_mixed_as_occupied=treat_mixed_as_occupied,
    )
    outer_grid = embed_inner_grid_with_walls(
        inner_grid=inner_grid,
        inner_domain=inner_domain,
        outer_domain=outer_domain,
        models=models,
        resolution=resolution,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pgm_path = output_stem.with_suffix(".pgm")
    yaml_path = output_stem.with_suffix(".yaml")

    export_grid_to_pgm(outer_grid, pgm_path)
    export_map_yaml(
        yaml_path=yaml_path,
        image_name=pgm_path.name,
        resolution=resolution,
        origin=[outer_domain.xmin, outer_domain.ymin, 0.0],
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
        negate=negate,
        mode=mode,
    )
    return pgm_path, yaml_path


def embed_inner_grid_with_walls(
    inner_grid: Sequence[Sequence[int]],
    inner_domain: AARect,
    outer_domain: AARect,
    models: Sequence[Model],
    resolution: float,
) -> list[list[int]]:
    # Rasterize the four outer walls before inserting the inner occupancy grid.
    width = int(round(outer_domain.width / resolution))
    height = int(round(outer_domain.height / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("The outer domain must contain at least one grid cell")

    grid = [[0 for _ in range(width)] for _ in range(height)]
    wall_obstacles = [model_to_obstacle(model) for model in models if model.type == "wall"]

    for iy in range(height):
        for ix in range(width):
            cell_rect = AARect(
                xmin=outer_domain.xmin + ix * resolution,
                ymin=outer_domain.ymin + iy * resolution,
                xmax=outer_domain.xmin + (ix + 1) * resolution,
                ymax=outer_domain.ymin + (iy + 1) * resolution,
            )
            if any(wall.intersects_rect(cell_rect) for wall in wall_obstacles):
                grid[iy][ix] = 1

    x_offset = int(round((inner_domain.xmin - outer_domain.xmin) / resolution))
    y_offset = int(round((inner_domain.ymin - outer_domain.ymin) / resolution))

    for iy, row in enumerate(inner_grid):
        target_y = iy + y_offset
        if target_y < 0 or target_y >= height:
            raise ValueError("The inner grid extends beyond the outer domain")
        for ix, value in enumerate(row):
            target_x = ix + x_offset
            if target_x < 0 or target_x >= width:
                raise ValueError("The inner grid extends beyond the outer domain")
            grid[target_y][target_x] = int(value)

    return grid


def export_grid_to_pgm(grid: Sequence[Sequence[int]], pgm_path: Path) -> None:
    # Write occupied cells as black and free cells as near-white in binary PGM format.
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if width <= 0 or any(len(row) != width for row in grid):
        raise ValueError("The occupancy grid must be a non-empty rectangle")

    pgm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pgm_path, "wb") as file:
        file.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        for row in reversed(grid):
            file.write(bytearray(0 if value == 1 else 254 for value in row))


def export_map_yaml(
    yaml_path: Path,
    image_name: str,
    resolution: float,
    origin: Sequence[float],
    occupied_thresh: float,
    free_thresh: float,
    negate: int,
    mode: str,
) -> None:
    # Emit the metadata format expected by Nav2 map_server.
    if len(origin) != 3:
        raise ValueError("origin must contain x, y, and yaw")

    payload = {
        "image": image_name,
        "mode": mode,
        "resolution": float(resolution),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "negate": int(negate),
        "occupied_thresh": float(occupied_thresh),
        "free_thresh": float(free_thresh),
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
