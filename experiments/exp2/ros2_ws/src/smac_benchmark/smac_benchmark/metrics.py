from __future__ import annotations

from math import cos, sin

import numpy as np

from smac_benchmark.world_loader import Obstacle


def remove_duplicate_points(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return path_xy
    keep = np.ones(len(path_xy), dtype=bool)
    keep[1:] = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1) > 0.0
    return path_xy[keep]


def compute_arc_length(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.array([], dtype=float)
    if len(path_xy) == 1:
        return np.array([0.0], dtype=float)
    seg_lengths = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def resample_polyline_by_arclength(path_xy: np.ndarray, spacing: float) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError("Resampling spacing must be positive.")
    path_xy = remove_duplicate_points(path_xy)
    s = compute_arc_length(path_xy)
    if len(s) < 2 or s[-1] == 0.0:
        return path_xy
    sample_s = np.arange(0.0, s[-1], spacing, dtype=float)
    if sample_s.size == 0 or sample_s[-1] < s[-1]:
        sample_s = np.append(sample_s, s[-1])
    sample_x = np.interp(sample_s, s, path_xy[:, 0])
    sample_y = np.interp(sample_s, s, path_xy[:, 1])
    return np.column_stack((sample_x, sample_y))


def compute_discrete_curvature(resampled_xy: np.ndarray, spacing: float) -> np.ndarray:
    if len(resampled_xy) < 3:
        return np.array([], dtype=float)
    diffs = np.diff(resampled_xy, axis=0)
    headings = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
    if len(headings) < 2:
        return np.array([], dtype=float)
    return np.diff(headings) / spacing


def compute_curvature_metrics(path_xy: np.ndarray, spacing: float) -> tuple[float, float]:
    resampled_xy = resample_polyline_by_arclength(path_xy, spacing)
    kappa = compute_discrete_curvature(resampled_xy, spacing)
    if kappa.size == 0:
        return 0.0, 0.0
    abs_kappa = np.abs(kappa)
    mean_abs_curvature = float(np.mean(abs_kappa))
    curvature_energy = float(np.mean(kappa ** 2))
    return mean_abs_curvature, curvature_energy


def point_to_box_clearance(point_xy: np.ndarray, obstacle: Obstacle) -> float:
    dx = point_xy[0] - obstacle.cx
    dy = point_xy[1] - obstacle.cy
    c = cos(-obstacle.yaw)
    s = sin(-obstacle.yaw)
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    qx = abs(local_x) - obstacle.hx
    qy = abs(local_y) - obstacle.hy
    outside_x = max(qx, 0.0)
    outside_y = max(qy, 0.0)
    outside_dist = float(np.hypot(outside_x, outside_y))
    inside_dist = min(max(qx, qy), 0.0)
    return outside_dist + inside_dist


def point_to_cylinder_clearance(point_xy: np.ndarray, obstacle: Obstacle) -> float:
    return float(np.hypot(point_xy[0] - obstacle.cx, point_xy[1] - obstacle.cy) - obstacle.r)


def point_to_obstacle_clearance(point_xy: np.ndarray, obstacle: Obstacle) -> float:
    if obstacle.kind == "box":
        return point_to_box_clearance(point_xy, obstacle)
    if obstacle.kind == "cylinder":
        return point_to_cylinder_clearance(point_xy, obstacle)
    raise ValueError(f"Unsupported obstacle kind: {obstacle.kind}")


def compute_clearance_metrics(
    path_xy: np.ndarray,
    obstacles: list[Obstacle],
    spacing: float,
    low_clearance_threshold_m: float,
) -> tuple[float, float]:
    if not obstacles:
        raise ValueError("No obstacles were loaded from the world file.")
    resampled_xy = resample_polyline_by_arclength(path_xy, spacing)
    if len(resampled_xy) == 0:
        raise ValueError("Resampled path is empty.")
    clearances = []
    for point_xy in resampled_xy:
        min_clearance = min(point_to_obstacle_clearance(point_xy, obs) for obs in obstacles)
        clearances.append(min_clearance)
    clearance_arr = np.array(clearances, dtype=float)
    mean_clearance = float(np.mean(clearance_arr))
    low_clearance_ratio = float(np.mean(clearance_arr < low_clearance_threshold_m))
    return mean_clearance, low_clearance_ratio


