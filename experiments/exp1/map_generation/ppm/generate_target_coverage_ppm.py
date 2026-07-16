from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_random_ppm import load_config, resolve_path
from modules.build_ppm import (
    build_ppm_polygons,
    build_sensor_poses,
    compute_bounds_from_obstacles,
    compute_inner_bounds_from_walls,
)
from modules.load_world import WorldLoader
from modules.plot_ppm import plot_proximity_with_ppm
from modules.prox_detector import ProxDetector, SensorProx
from modules.sensor_random import sample_sensor_positions
from modules.build_ppm import save_ppm, build_prox_pointset


def parse_args() -> argparse.Namespace:
    # Parse CLI args for target-coverage reconstruction from an existing area CSV.
    parser = argparse.ArgumentParser(
        description="Generate only the first PPM that exceeds a target coverage ratio."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--coverage-ratio",
        type=float,
        nargs="+",
        default=None,
        help="One or more coverage ratio targets. Values greater than 1 are treated as percentages.",
    )
    parser.add_argument(
        "--area-csv",
        default=None,
        help="Existing ppm_area.csv used to choose the required number of sensors.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for ppm_sensor.csv, ppm_prox.csv, and the target PPM plot.",
    )
    return parser.parse_args()


def normalize_coverage_ratio(value: float) -> float:
    # Accept both ratio inputs (0.8) and percentage inputs (80).
    normalized = value / 100.0 if value > 1.0 else value
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"coverage ratio must be in (0, 1] or (0, 100], got {value}")
    return normalized


def normalize_coverage_ratios(value: float | list[float] | tuple[float, ...]) -> list[float]:
    # Normalize scalar or list inputs and keep the requested order without duplicates.
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    normalized_values: list[float] = []
    for raw_value in raw_values:
        normalized = normalize_coverage_ratio(float(raw_value))
        if normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def load_target_sensor_count(area_csv_path: Path, target_ratio: float) -> tuple[int, float]:
    # Pick the first sensor count whose coverage meets or exceeds the requested ratio.
    with area_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"num_sensors", "coverage_ratio"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{area_csv_path} must contain columns {sorted(required_columns)}."
            )

        for row in reader:
            coverage_ratio = float(row["coverage_ratio"])
            if coverage_ratio >= target_ratio:
                return int(row["num_sensors"]), coverage_ratio

    raise ValueError(
        f"No entry in {area_csv_path} reaches target coverage ratio {target_ratio:.4f}."
    )


def build_output_config(config: dict, out_dir: Path) -> dict:
    # Redirect CSV outputs to an isolated directory for target-only generation.
    out_dir.mkdir(parents=True, exist_ok=True)
    updated_config = dict(config)
    eval_cfg = dict(updated_config.get("evaluation", {}))
    eval_cfg["out_sensor_csv"] = str(out_dir / "ppm_sensor.csv")
    eval_cfg["out_prox_csv"] = str(out_dir / "ppm_prox.csv")
    updated_config["evaluation"] = eval_cfg
    return updated_config


def resolve_output_dir(
    args: argparse.Namespace,
    target_cfg: dict,
    config_path: Path,
    repo_root: Path,
    area_csv_path: Path,
    target_ratio: float,
    multiple_targets: bool,
) -> Path:
    # Resolve output directory while avoiding collisions across multiple targets.
    percent_label = int(round(target_ratio * 100))
    if args.out_dir is not None:
        cli_out_dir = Path(args.out_dir).resolve()
        if multiple_targets:
            return cli_out_dir / f"coverage_{percent_label:02d}"
        return cli_out_dir

    template_value = target_cfg.get("out_dir_template")
    if template_value:
        formatted = str(template_value).format(
            percent=percent_label,
            ratio=f"{target_ratio:.4f}",
        )
        return resolve_path(formatted, base_dir=config_path.parent, fallback_dir=repo_root)

    return area_csv_path.parent / f"coverage_{percent_label:02d}"


def generate_target_ppm(
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    area_csv_path: Path,
    target_ratio: float,
    out_dir: Path,
) -> None:
    # Generate target-specific PPM artifacts for the first sensor count that reaches the ratio.
    target_sensor_count, achieved_ratio = load_target_sensor_count(area_csv_path, target_ratio)

    print(f"[INFO] Target coverage ratio: {target_ratio:.4f}")
    print(f"[INFO] Source area CSV: {area_csv_path}")
    print(f"[INFO] Selected sensor count: {target_sensor_count} (coverage_ratio={achieved_ratio:.6f})")
    print(f"[INFO] Output directory: {out_dir}")

    world_file = resolve_path(
        config["io"]["world_file"],
        base_dir=config_path.parent,
        fallback_dir=repo_root,
    )
    loader = WorldLoader(world_file=str(world_file))
    obstacles = loader.load_obstacles()
    print(f"[INFO] World loaded from {world_file} with {len(obstacles)} obstacles.")

    sampling_min, sampling_max = compute_inner_bounds_from_walls(obstacles, padding=0.0)
    print(
        f"[INFO] Sampling bounds: x[{sampling_min[0]:.2f}, {sampling_max[0]:.2f}], "
        f"y[{sampling_min[1]:.2f}, {sampling_max[1]:.2f}]"
    )

    sensor_cfg = config["sensor"]
    sensor_positions = sample_sensor_positions(
        obstacles=obstacles,
        num_samples=target_sensor_count,
        x_range=(sampling_min[0], sampling_max[0]),
        y_range=(sampling_min[1], sampling_max[1]),
        seed=int(sensor_cfg.get("random_seed", 0)),
        max_range_m=float(sensor_cfg["max_range_m"]),
        fov_deg=float(sensor_cfg["angle_deg"]),
        sampling_method=str(sensor_cfg.get("sampling_method", "random")),
        halton_scramble=bool(sensor_cfg.get("halton_scramble", True)),
        angle_step_deg=2.0,
    )
    if len(sensor_positions) != target_sensor_count:
        raise RuntimeError(
            f"Expected {target_sensor_count} sampled sensors, but got {len(sensor_positions)}."
        )
    print(f"[INFO] Re-sampled {len(sensor_positions)} sensor positions.")

    detector = ProxDetector(obstacles=obstacles)
    sensor_prox: list[SensorProx] = []
    for count, sensor in enumerate(sensor_positions):
        prox = detector.detect(
            sensor_xyyaw=sensor,
            max_range_m=float(sensor_cfg["max_range_m"]),
            fov_deg=float(sensor_cfg["angle_deg"]),
            angle_step_deg=2.0,
        )
        sensor_prox.append(SensorProx(count=count, sensor_xyyaw=sensor, prox=prox))
        if (count + 1) % 10 == 0 or (count + 1) == len(sensor_positions):
            print(f"[INFO] Detected proximity points from sensor {count + 1}/{len(sensor_positions)}.")

    sensors = build_sensor_poses(sensor_positions)
    bounds_min, bounds_max = compute_bounds_from_obstacles(obstacles, padding=1.0)
    ppm_polygons = build_ppm_polygons(
        sensors=sensors,
        prox_points=build_prox_pointset(sensor_prox),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )

    output_config = build_output_config(config, out_dir)
    save_ppm(sensor_prox=sensor_prox, config=output_config)

    plot_path = out_dir / "ppm_target_plot.png"
    plot_proximity_with_ppm(
        obstacles=obstacles,
        sensor_prox=sensor_prox,
        ppm_polygons=ppm_polygons,
        show=False,
        output_path=str(plot_path),
        formats=["png", "eps"],
    )
    print(f"[INFO] Saved target PPM plot to {plot_path}.")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]

    config = load_config(str(config_path))
    target_cfg = config.get("target", {})

    target_ratio_raw = (
        args.coverage_ratio
        if args.coverage_ratio is not None
        else target_cfg.get("coverage_ratios")
    )
    if target_ratio_raw is None:
        raise ValueError("target coverage ratio is not set. Use --coverage-ratio or target.coverage_ratios.")
    target_ratios = normalize_coverage_ratios(target_ratio_raw)

    area_csv_value = args.area_csv if args.area_csv is not None else config.get("io", {}).get("area_csv")
    if not area_csv_value:
        raise ValueError("target area CSV is not set. Use --area-csv or io.area_csv.")
    area_csv_path = resolve_path(str(area_csv_value), base_dir=config_path.parent, fallback_dir=repo_root)
    if not area_csv_path.exists():
        raise FileNotFoundError(f"area CSV not found: {area_csv_path}")

    multiple_targets = len(target_ratios) > 1
    for target_ratio in target_ratios:
        out_dir = resolve_output_dir(
            args=args,
            target_cfg=target_cfg,
            config_path=config_path,
            repo_root=repo_root,
            area_csv_path=area_csv_path,
            target_ratio=target_ratio,
            multiple_targets=multiple_targets,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_target_ppm(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            area_csv_path=area_csv_path,
            target_ratio=target_ratio,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()

