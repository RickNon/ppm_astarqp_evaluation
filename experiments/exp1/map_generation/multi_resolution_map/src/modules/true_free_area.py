from __future__ import annotations

import math

from .geometry2d import AARect
from .obstacles import BoxObstacle2D, CylinderObstacle2D, Obstacle2D


def compute_true_free_area(obstacles: list[Obstacle2D], bounds: AARect) -> float:
    # Match random_ppm_generator by subtracting exact obstacle area from the inner wall bounds.
    try:
        from shapely.affinity import rotate as rotate_geom
        from shapely.geometry import Point
        from shapely.geometry import box as shapely_box
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Shapely is required for exact free-area computation.") from exc

    if bounds.width <= 0.0 or bounds.height <= 0.0:
        return 0.0

    bounds_poly = shapely_box(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)

    obstacle_geoms = []
    for obstacle in obstacles:
        if isinstance(obstacle, BoxObstacle2D):
            shape = obstacle.shape
            rect = shapely_box(
                shape.center.x - shape.width * 0.5,
                shape.center.y - shape.height * 0.5,
                shape.center.x + shape.width * 0.5,
                shape.center.y + shape.height * 0.5,
            )
            if shape.yaw != 0.0:
                rect = rotate_geom(
                    rect,
                    math.degrees(shape.yaw),
                    origin=(shape.center.x, shape.center.y),
                )
            obstacle_geoms.append(rect)
        elif isinstance(obstacle, CylinderObstacle2D):
            shape = obstacle.shape
            obstacle_geoms.append(
                Point(shape.center.x, shape.center.y).buffer(shape.radius, resolution=64)
            )
        else:
            raise TypeError(f"Unsupported obstacle type: {type(obstacle)}")

    if not obstacle_geoms:
        return bounds_poly.area

    union_obs = unary_union(obstacle_geoms)
    occupied = bounds_poly.intersection(union_obs).area
    return max(bounds_poly.area - occupied, 0.0)
