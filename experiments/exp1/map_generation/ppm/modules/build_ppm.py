from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.load_world import Obstacle, is_wall_obstacle_name
from modules.prox_detector import SensorProx

# Keep ppm_builder import path local to this module.
_ROOT = Path(__file__).resolve().parent
_PPM_BUILDER = _ROOT / "Polygonal-Proximity-Map" / "ppm_builder"
if str(_PPM_BUILDER) not in sys.path:
    sys.path.append(str(_PPM_BUILDER))

from ppm_builder import PPMBuilder2D, SensorPose, ProximityPointSet
from ppm_builder.export.halfspace_extract import extract_vertices


def _build_eval_indices(
    total_sensors: int,
    dense_until: int = 100,
    growth: float = 1.2,
) -> list[int]:
    # Use dense sampling early, then widen intervals with a geometric schedule.
    if total_sensors <= 0:
        return []

    dense_end = min(dense_until, total_sensors)
    indices = list(range(1, dense_end + 1))
    if total_sensors <= dense_end:
        return indices

    current = dense_end
    while current < total_sensors:
        next_k = int(math.ceil(current * growth))
        if next_k <= current:
            next_k = current + 1
        current = min(next_k, total_sensors)
        if current != indices[-1]:
            indices.append(current)

    return indices


def build_sensor_poses(
    sensor_positions: Sequence[tuple[float, float, float]]
) -> list[SensorPose]:
    return [
        SensorPose(id=i, position=np.array([s[0], s[1]], dtype=np.float64))
        for i, s in enumerate(sensor_positions)
    ]


def build_prox_pointset(
    sensor_prox: Sequence[SensorProx],
    upto_count: int | None = None,
) -> ProximityPointSet:
    prox_slice = sensor_prox if upto_count is None else sensor_prox[:upto_count]
    prox_dict = {sp.count: [(h.qx, h.qy) for h in sp.prox] for sp in prox_slice}
    return ProximityPointSet.from_dict(prox_dict)


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    # Shoelace formula for polygon area.
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def build_ppm_polygons(
    sensors: Sequence[SensorPose],
    prox_points: ProximityPointSet,
    bounds_min: tuple[float, float],
    bounds_max: tuple[float, float],
) -> list[list[tuple[float, float]]]:
    builder = PPMBuilder2D()
    cells = builder.build_halfspaces(
        sensors=sensors,
        prox_points=prox_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        eps=1e-3,
        include_aabb=True,
    )
    ppm_polygons: list[list[tuple[float, float]]] = []
    for cell in cells.values():
        extracted = extract_vertices(cell.halfspaces)
        if extracted is None:
            continue
        verts = extracted.vertices
        if extracted.hull is not None:
            verts = verts[extracted.hull.vertices]
        ppm_polygons.append([(float(x), float(y)) for x, y in verts])
    return ppm_polygons


def compute_bounds_from_obstacles(
    obstacles: Iterable[Obstacle],
    padding: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = []
    ys = []
    for o in obstacles:
        if o.kind == "box":
            xs.extend([o.cx - o.hx, o.cx + o.hx])
            ys.extend([o.cy - o.hy, o.cy + o.hy])
        else:
            xs.extend([o.cx - o.r, o.cx + o.r])
            ys.extend([o.cy - o.r, o.cy + o.r])
    bounds_min = (min(xs) - padding, min(ys) - padding)
    bounds_max = (max(xs) + padding, max(ys) + padding)
    return bounds_min, bounds_max


def compute_inner_bounds_from_walls(
    obstacles: Iterable[Obstacle],
    padding: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    # Use the inner faces of enclosure wall boxes as sampling bounds.
    walls = [o for o in obstacles if is_wall_obstacle_name(o.name) and o.kind == "box"]
    if not walls:
        # Fallback to the full obstacle bounds when no walls are present.
        return compute_bounds_from_obstacles(obstacles, padding=padding)

    inner_min_x = min(o.cx + o.hx for o in walls)
    inner_max_x = max(o.cx - o.hx for o in walls)
    inner_min_y = min(o.cy + o.hy for o in walls)
    inner_max_y = max(o.cy - o.hy for o in walls)

    if inner_min_x >= inner_max_x or inner_min_y >= inner_max_y:
        # Fallback when walls do not form a valid enclosing rectangle.
        return compute_bounds_from_obstacles(obstacles, padding=padding)

    bounds_min = (inner_min_x + padding, inner_min_y + padding)
    bounds_max = (inner_max_x - padding, inner_max_y - padding)
    return bounds_min, bounds_max


def compute_true_free_area(
    obstacles: Iterable[Obstacle],
    bounds_min: tuple[float, float],
    bounds_max: tuple[float, float],
) -> float:
    # Compute free area exactly using polygon boolean operations (requires shapely).
    try:
        from shapely.geometry import box as shapely_box
        from shapely.geometry import Point
        from shapely.affinity import rotate as rotate_geom
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Shapely is required for exact free-area computation.") from exc

    min_x, min_y = bounds_min
    max_x, max_y = bounds_max
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0.0 or height <= 0.0:
        return 0.0

    bounds_poly = shapely_box(min_x, min_y, max_x, max_y)

    def _box_polygon(o: Obstacle):
        # Build a rotated rectangle polygon for the obstacle footprint.
        rect = shapely_box(o.cx - o.hx, o.cy - o.hy, o.cx + o.hx, o.cy + o.hy)
        if o.yaw == 0.0:
            return rect
        return rotate_geom(rect, math.degrees(o.yaw), origin=(o.cx, o.cy))

    def _cylinder_polygon(o: Obstacle):
        # Use a high-resolution buffer for accurate circle approximation.
        return Point(o.cx, o.cy).buffer(o.r, resolution=64)

    obstacle_geoms = []
    for o in obstacles:
        # Include wall_ blocks for mazes so passages correctly subtract occupied cells.
        if o.kind == "box":
            obstacle_geoms.append(_box_polygon(o))
        else:
            obstacle_geoms.append(_cylinder_polygon(o))

    if not obstacle_geoms:
        return bounds_poly.area

    union_obs = unary_union(obstacle_geoms)
    occupied = bounds_poly.intersection(union_obs).area
    return max(bounds_poly.area - occupied, 0.0)


def evaluate_ppm_area_records(
    sensors: Sequence[SensorPose],
    sensor_prox: Sequence[SensorProx],
    bounds_min: tuple[float, float],
    bounds_max: tuple[float, float],
    true_free_area: float,
    eval_indices: Sequence[int] | None = None,
) -> list[tuple[int, float, float]]:
    area_records: list[tuple[int, float, float]] = []
    if eval_indices is None:
        # Evaluate every iteration when no explicit indices are provided.
        eval_indices = list(range(1, len(sensors) + 1))
    else:
        # Normalize evaluation indices to valid, unique, ascending values.
        eval_indices = sorted({k for k in eval_indices if 1 <= k <= len(sensors)})
    for k in eval_indices:
        prox_points = build_prox_pointset(sensor_prox, upto_count=k)
        ppm_polygons = build_ppm_polygons(
            sensors=sensors[:k],
            prox_points=prox_points,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        ppm_area = sum(polygon_area(poly) for poly in ppm_polygons)
        ratio = ppm_area / true_free_area if true_free_area > 0.0 else 0.0
        area_records.append((k, ppm_area, ratio))
        if k % 10 == 0 or k == eval_indices[-1]:
            print(f"[INFO] Evaluated PPM area: {k}/{len(sensors)}.")
    return area_records

def write_area_csv(area_records: Sequence[tuple[int, float, float]], csv_path: str) -> None:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_sensors', 'ppm_area', 'coverage_ratio'])
        for row in area_records:
            writer.writerow(row)


def plot_area_records(area_records: Sequence[tuple[int, float, float]], output_path: str) -> None:
    xs = [r[0] for r in area_records]
    ys = [r[2] for r in area_records]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker='o', color='#1f77b4')
    ax.set_xlabel('Number of sensors')
    ax.set_ylabel('PPM free area coverage ratio')
    ax.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_ppm(sensor_prox: Sequence[SensorProx], config: dict) -> None:
    io_cfg = config.get("evaluation", {})
    sensor_csv = io_cfg.get("out_sensor_csv")
    prox_csv = io_cfg.get("out_prox_csv")
    if not sensor_csv or not prox_csv:
        raise ValueError("Missing io.out_sensor_csv or io.out_prox_csv in config.")

    sensor_path = Path(sensor_csv)
    prox_path = Path(prox_csv)
    sensor_path.parent.mkdir(parents=True, exist_ok=True)
    prox_path.parent.mkdir(parents=True, exist_ok=True)

    with sensor_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count",
            "time",
            "location_odometry_pos_x",
            "location_odometry_pos_y",
            "location_odometry_pos_z",
            "location_odometry_roll",
            "location_odometry_pitch",
            "location_odometry_yaw",
        ])
        for sp in sensor_prox:
            count = sp.count
            time = count * 500
            sx, sy, syaw = sp.sensor_xyyaw
            writer.writerow([count, time, sx, sy, 0, 0, 0, syaw])

    with prox_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count",
            "time",
            "proximity_num",
            "proximity_id",
            "proximity_duration",
            "proximity_velocity",
            "proximity_theta",
            "proximity_phi",
            "proximity_distance",
        ])
        for sp in sensor_prox:
            count = sp.count
            time = count * 500
            sx, sy, _ = sp.sensor_xyyaw
            prox_num = len(sp.prox)

            row = [count, time, prox_num]
            for i, hit in enumerate(sp.prox):
                # Convert relative (dx, dy) to spherical with z=0.
                dx = hit.qx - sx
                dy = hit.qy - sy
                distance = math.hypot(dx, dy)
                theta = math.atan2(dy, dx)  # azimuth
                phi = 0.0                   # elevation, z=0
                row.extend([i, 0, 0, theta, phi, distance])
            writer.writerow(row)

