#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import rclpy

from smac_benchmark.batch_io import write_path_csv, write_results, write_start_goal_pairs
from smac_benchmark.batch_config import load_batch_benchmark_config
from smac_benchmark.metrics import compute_clearance_metrics, compute_curvature_metrics
from smac_benchmark.plotter import save_trial_plot
from smac_benchmark.run_benchmark import (
    BenchmarkNode,
    BenchmarkRunConfig,
    run_benchmark_once,
    write_benchmark_outputs,
)
from smac_benchmark.world_loader import WorldLoader


def _read_start_goal_pairs(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "count",
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
                "success",
                "status",
                "planning_time_sec",
                "path_waypoint_count",
                "path_length_m",
                "selected_start_yaw_rad",
                "selected_goal_yaw_rad",
                "result_txt",
                "result_csv",
                "result_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _payload_path_xy(payload: dict[str, object]) -> np.ndarray:
    points = payload.get("path_waypoints", [])
    if not isinstance(points, list) or not points:
        return np.empty((0, 2), dtype=float)
    return np.array(
        [[float(point["x"]), float(point["y"])] for point in points],
        dtype=float,
    )


def _to_tmp_result_row(
    pair: dict[str, str],
    payload: dict[str, object],
    path_xy: np.ndarray,
    obstacles: list[object],
    resample_spacing_m: float,
    low_clearance_threshold_m: float,
) -> dict[str, object]:
    success = bool(payload["success"])
    row: dict[str, object] = {
        "count": int(pair["count"]),
        "start_x": float(pair["start_x"]),
        "start_y": float(pair["start_y"]),
        "goal_x": float(pair["goal_x"]),
        "goal_y": float(pair["goal_y"]),
        "success": 1 if success else 0,
        "time": float(payload["planning_time_sec"]) * 1000.0,
        "path_length": "",
        "mean_abs_curvature": "",
        "curvature_energy": "",
        "mean_clearance": "",
        "low_clearance_ratio": "",
    }

    if not success or len(path_xy) == 0:
        return row

    row["path_length"] = float(payload["path_length_m"])
    mean_abs_curvature, curvature_energy = compute_curvature_metrics(
        path_xy,
        spacing=resample_spacing_m,
    )
    row["mean_abs_curvature"] = mean_abs_curvature
    row["curvature_energy"] = curvature_energy

    if obstacles:
        mean_clearance, low_clearance_ratio = compute_clearance_metrics(
            path_xy,
            obstacles=obstacles,
            spacing=resample_spacing_m,
            low_clearance_threshold_m=low_clearance_threshold_m,
        )
        row["mean_clearance"] = mean_clearance
        row["low_clearance_ratio"] = low_clearance_ratio

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a final experiment config YAML.",
    )
    args = parser.parse_args()

    config = load_batch_benchmark_config(Path(args.config))
    pairs = _read_start_goal_pairs(config.start_goal_csv)
    obstacles = WorldLoader(config.world_file).load_obstacles() if config.world_file is not None else []

    output_dir = config.output_dir
    method_output_dir = output_dir / config.method_name
    raw_output_dir = method_output_dir / config.raw_output_dirname
    method_paths_dir = method_output_dir / "paths"
    method_plots_dir = method_output_dir / config.plots_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    method_output_dir.mkdir(parents=True, exist_ok=True)
    method_paths_dir.mkdir(parents=True, exist_ok=True)
    if config.save_tmp_compatible_outputs and config.save_tmp_compatible_plots:
        method_plots_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config.config_path, method_output_dir / "config_used.yml")
    shutil.copyfile(config.start_goal_csv, output_dir / "start_goal_pairs.csv")
    if config.save_tmp_compatible_outputs:
        write_start_goal_pairs(method_output_dir / "start_goal_pairs.csv", [dict(pair) for pair in pairs])

    rclpy.init()
    node = BenchmarkNode()

    summary_rows: list[dict[str, object]] = []
    tmp_result_rows: list[dict[str, object]] = []
    total_pairs = len(pairs)

    try:
        for index, pair in enumerate(pairs, start=1):
            count = str(pair["count"])
            node.get_logger().info(f"[{index}/{total_pairs}] Planning pair count={count}")

            run_config = BenchmarkRunConfig(
                start_xy=(float(pair["start_x"]), float(pair["start_y"])),
                goal_xy=(float(pair["goal_x"]), float(pair["goal_y"])),
                frame_id=config.frame,
                timeout_sec=config.timeout_sec,
                start_yaw=config.start_yaw,
                goal_yaw=config.goal_yaw,
                auto_start_yaw=config.auto_start_yaw,
                auto_goal_yaw=config.auto_goal_yaw,
                yaw_samples=config.yaw_samples,
                max_planning_attempts=config.max_planning_attempts,
                planner_cmd_substr=config.planner_cmd_substr,
            )
            result = run_benchmark_once(node=node, config=run_config)

            out_base = raw_output_dir / count
            write_benchmark_outputs(
                node=node,
                result=result,
                out_base=str(out_base),
                map_yaml_path=str(config.map_yaml),
                save_plot=config.save_individual_plots,
            )

            payload = result.payload
            path_xy = _payload_path_xy(payload)
            relative_out_base = out_base.relative_to(method_output_dir)
            summary_rows.append(
                {
                    "count": count,
                    "start_x": pair["start_x"],
                    "start_y": pair["start_y"],
                    "goal_x": pair["goal_x"],
                    "goal_y": pair["goal_y"],
                    "success": payload["success"],
                    "status": payload["status"],
                    "planning_time_sec": payload["planning_time_sec"],
                    "path_waypoint_count": payload["path_waypoint_count"],
                    "path_length_m": payload["path_length_m"],
                    "selected_start_yaw_rad": payload["selected_start_yaw_rad"],
                    "selected_goal_yaw_rad": payload["selected_goal_yaw_rad"],
                    "result_txt": str(relative_out_base.with_suffix(".txt")),
                    "result_csv": str(relative_out_base.with_suffix(".csv")),
                    "result_png": str(relative_out_base.with_suffix(".png")) if config.save_individual_plots else "",
                }
            )

            if config.save_tmp_compatible_outputs:
                tmp_result_rows.append(
                    _to_tmp_result_row(
                        pair=pair,
                        payload=payload,
                        path_xy=path_xy,
                        obstacles=obstacles,
                        resample_spacing_m=config.resample_spacing_m,
                        low_clearance_threshold_m=config.low_clearance_threshold_m,
                    )
                )
                if bool(payload["success"]) and len(path_xy) > 0:
                    write_path_csv(
                        output_path=method_paths_dir / f"{count}_path.csv",
                        count=int(pair["count"]),
                        path_xy=path_xy,
                    )
                if config.save_tmp_compatible_plots:
                    save_trial_plot(
                        output_path=method_plots_dir / f"{count}_plot.png",
                        obstacles=obstacles,
                        free_area_polygons=[],
                        start=(float(pair["start_x"]), float(pair["start_y"])),
                        goal=(float(pair["goal_x"]), float(pair["goal_y"])),
                        method_paths={config.method_name: path_xy if len(path_xy) > 0 else None},
                        title=f"Trial {count}",
                    )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    _write_summary_csv(method_output_dir / config.summary_csv_name, summary_rows)
    if config.save_tmp_compatible_outputs:
        write_results(method_output_dir / "results.csv", tmp_result_rows)


if __name__ == "__main__":
    main()

