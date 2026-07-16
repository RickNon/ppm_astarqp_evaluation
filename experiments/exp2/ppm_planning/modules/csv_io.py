from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_start_goal_pairs(input_path: Path) -> list[dict[str, float | int]]:
    """Read fixed start-goal pairs from a previously saved experiment CSV."""
    with input_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_fields = {"count", "start_x", "start_y", "goal_x", "goal_y"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"{input_path} must contain columns: {sorted(required_fields)}")

        pairs: list[dict[str, float | int]] = []
        for row in reader:
            pairs.append(
                {
                    "count": int(float(row["count"])),
                    "start_x": float(row["start_x"]),
                    "start_y": float(row["start_y"]),
                    "goal_x": float(row["goal_x"]),
                    "goal_y": float(row["goal_y"]),
                }
            )
    return pairs


def write_start_goal_pairs(output_path: Path, pairs: list[dict[str, float | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["count", "start_x", "start_y", "goal_x", "goal_y"],
        )
        writer.writeheader()
        writer.writerows(pairs)


def write_results(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "count",
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
                "success",
                "time",
                "path_length",
                "mean_abs_curvature",
                "curvature_energy",
                "mean_clearance",
                "low_clearance_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_debug_results(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["count", "success", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _compute_yaw_values(path_xy: np.ndarray) -> np.ndarray:
    """Compute per-point yaw from the local path tangent."""
    if len(path_xy) == 0:
        return np.array([], dtype=float)
    if len(path_xy) == 1:
        return np.array([0.0], dtype=float)

    yaw_values = np.zeros(len(path_xy), dtype=float)
    diffs = np.diff(path_xy, axis=0)
    segment_yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
    yaw_values[:-1] = segment_yaw
    yaw_values[-1] = segment_yaw[-1]
    return yaw_values


def write_path_csv(output_path: Path, count: int, path_xy: np.ndarray) -> None:
    """Write one successful path as count,x,y,yaw CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaw_values = _compute_yaw_values(path_xy)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["count", "x", "y", "yaw"])
        for point_xy, yaw in zip(path_xy, yaw_values):
            writer.writerow([count, float(point_xy[0]), float(point_xy[1]), float(yaw)])


def clear_matching_files(directory: Path, pattern: str) -> None:
    """Remove previously generated files so reruns do not leave stale artifacts."""
    if not directory.exists():
        return
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()

