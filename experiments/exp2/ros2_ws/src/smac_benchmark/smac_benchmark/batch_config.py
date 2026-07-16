from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BatchBenchmarkConfig:
    config_path: Path
    map_yaml: Path
    start_goal_csv: Path
    output_dir: Path
    world_file: Path | None
    planner_params: Path | None
    costmap_params: Path | None
    frame: str
    timeout_sec: float
    start_yaw: float
    goal_yaw: float
    auto_start_yaw: bool
    auto_goal_yaw: bool
    yaw_samples: int
    max_planning_attempts: int | None
    planner_cmd_substr: str
    save_individual_plots: bool
    raw_output_dirname: str
    summary_csv_name: str
    method_name: str
    save_tmp_compatible_outputs: bool
    save_tmp_compatible_plots: bool
    plots_dirname: str
    resample_spacing_m: float
    low_clearance_threshold_m: float


def _resolve_path(raw_path: str | None, config_dir: Path) -> Path | None:
    if raw_path is None or raw_path == "":
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()

    direct_path = (config_dir / candidate).resolve()
    if direct_path.exists():
        return direct_path

    anchor_name = candidate.parts[0] if candidate.parts else ""
    if anchor_name:
        for parent in config_dir.resolve().parents:
            if (parent / anchor_name).exists():
                return (parent / candidate).resolve()

    for parent in config_dir.resolve().parents:
        parent_candidate = (parent / candidate).resolve()
        if parent_candidate.exists() or parent_candidate.parent.exists():
            return parent_candidate

    return direct_path


def load_batch_benchmark_config(config_path: Path) -> BatchBenchmarkConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    root = data.get("smac_benchmark", {})
    io_cfg = root.get("io", {})
    nav2_cfg = root.get("nav2", {})
    planning_cfg = root.get("planning", {})
    analysis_cfg = root.get("analysis", {})
    output_cfg = root.get("output", {})
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    return BatchBenchmarkConfig(
        config_path=config_path,
        map_yaml=_resolve_path(io_cfg["map_yaml"], config_dir),
        start_goal_csv=_resolve_path(io_cfg["start_goal_csv"], config_dir),
        output_dir=_resolve_path(io_cfg["output_dir"], config_dir),
        world_file=_resolve_path(io_cfg.get("world_file"), config_dir),
        planner_params=_resolve_path(nav2_cfg.get("planner_params"), config_dir),
        costmap_params=_resolve_path(nav2_cfg.get("costmap_params"), config_dir),
        frame=str(planning_cfg.get("frame", "map")),
        timeout_sec=float(planning_cfg.get("timeout_sec", 20.0)),
        start_yaw=float(planning_cfg.get("start_yaw", 0.0)),
        goal_yaw=float(planning_cfg.get("goal_yaw", 0.0)),
        auto_start_yaw=bool(planning_cfg.get("auto_start_yaw", False)),
        auto_goal_yaw=bool(planning_cfg.get("auto_goal_yaw", False)),
        yaw_samples=int(planning_cfg.get("yaw_samples", 16)),
        max_planning_attempts=(
            int(planning_cfg["max_planning_attempts"])
            if int(planning_cfg.get("max_planning_attempts", 0)) > 0
            else None
        ),
        planner_cmd_substr=str(nav2_cfg.get("planner_cmd_substr", "planner_server")),
        save_individual_plots=bool(output_cfg.get("save_individual_plots", True)),
        raw_output_dirname=str(output_cfg.get("raw_output_dirname", "raw")),
        summary_csv_name=str(output_cfg.get("summary_csv_name", "run_summary.csv")),
        method_name=str(output_cfg.get("method_name", "nav2_hybrid_astar")),
        save_tmp_compatible_outputs=bool(output_cfg.get("save_tmp_compatible_outputs", True)),
        save_tmp_compatible_plots=bool(output_cfg.get("save_tmp_compatible_plots", True)),
        plots_dirname=str(output_cfg.get("plots_dirname", "plots")),
        resample_spacing_m=float(analysis_cfg.get("resample_spacing_m", 0.1)),
        low_clearance_threshold_m=float(analysis_cfg.get("low_clearance_threshold_m", 0.5)),
    )


