from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    ppm_sensor_csv: Path
    ppm_prox_csv: Path
    world_file: Path | None
    output_dir: Path
    start_goal_pairs_csv: Path | None
    random_seed: int
    num_trials: int
    planner_methods: list[str]
    qp_solver: str
    smoothing_mu: float
    bitstar_time_limit_s: float
    bitstar_range_m: float | None
    bitstar_random_seed: int | None
    bitstar_simplify_solution: bool
    bitstar_state_validity_resolution: float | None
    min_start_goal_distance: float
    max_sampling_attempts: int
    bounds_margin_m: float
    resample_spacing_m: float
    low_clearance_threshold_m: float
    plot_enable: bool
    plot_output_dir: Path


def _resolve_path(raw_path: str, config_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_dir / candidate).resolve()


def _resolve_optional_path(raw_path: object, config_dir: Path) -> Path | None:
    if raw_path is None or raw_path == "":
        return None
    return _resolve_path(str(raw_path), config_dir)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    io_cfg = data.get("io", {})
    experiment_cfg = data.get("experiment", {})
    planner_cfg = data.get("planner", {})
    bitstar_cfg = data.get("bitstar_ppm", {})
    sampling_cfg = data.get("sampling", {})
    metrics_cfg = data.get("metrics", {})
    plot_cfg = data.get("plot", {})
    config_dir = config_path.resolve().parent
    planner_methods = planner_cfg.get("methods", ["astar_qp_base"])
    if isinstance(planner_methods, str):
        planner_methods = [planner_methods]

    return AppConfig(
        ppm_sensor_csv=_resolve_path(io_cfg["ppm_sensor_csv"], config_dir),
        ppm_prox_csv=_resolve_path(io_cfg["ppm_prox_csv"], config_dir),
        world_file=_resolve_optional_path(io_cfg.get("world_file"), config_dir),
        output_dir=_resolve_path(io_cfg["output_dir"], config_dir),
        start_goal_pairs_csv=_resolve_optional_path(io_cfg.get("start_goal_pairs_csv"), config_dir),
        random_seed=int(experiment_cfg.get("random_seed", 0)),
        num_trials=int(experiment_cfg.get("num_trials", 1)),
        planner_methods=[str(method) for method in planner_methods],
        qp_solver=str(planner_cfg.get("qp_solver", "OSQP")).upper(),
        smoothing_mu=float(planner_cfg.get("smoothing_mu", 1.0)),
        bitstar_time_limit_s=float(bitstar_cfg.get("time_limit_s", 0.2)),
        bitstar_range_m=_optional_float(bitstar_cfg.get("range_m")),
        bitstar_random_seed=_optional_int(bitstar_cfg.get("random_seed")),
        bitstar_simplify_solution=bool(bitstar_cfg.get("simplify_solution", True)),
        bitstar_state_validity_resolution=_optional_float(
            bitstar_cfg.get("state_validity_resolution")
        ),
        min_start_goal_distance=float(sampling_cfg.get("min_start_goal_distance", 0.0)),
        max_sampling_attempts=int(sampling_cfg.get("max_sampling_attempts", 10000)),
        bounds_margin_m=float(sampling_cfg.get("bounds_margin_m", 0.0)),
        resample_spacing_m=float(metrics_cfg.get("resample_spacing_m", 0.1)),
        low_clearance_threshold_m=float(metrics_cfg.get("low_clearance_threshold_m", 0.5)),
        plot_enable=bool(plot_cfg.get("enable", False)),
        plot_output_dir=_resolve_path(str(plot_cfg.get("output_dir", io_cfg["output_dir"])), config_dir),
    )

