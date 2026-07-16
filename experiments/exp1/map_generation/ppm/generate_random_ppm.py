from modules.load_world import WorldLoader
from modules.plot_ppm import plot_proximity_with_ppm
from modules.prox_detector import ProxDetector, SensorProx
from modules.build_ppm import (
    build_ppm_polygons,
    build_prox_pointset,
    build_sensor_poses,
    compute_bounds_from_obstacles,
    compute_inner_bounds_from_walls,
    compute_true_free_area,
    evaluate_ppm_area_records,
    plot_area_records,
    write_area_csv,
    save_ppm,
)

from pathlib import Path
import argparse
import yaml


def resolve_path(path_str: str, base_dir: Path, fallback_dir: Path | None = None) -> Path:
    # Resolve a path against the config directory first, then an optional fallback root.
    path = Path(path_str)
    if path.is_absolute():
        return path

    primary = (base_dir / path).resolve()
    if primary.exists():
        return primary

    if fallback_dir is not None:
        fallback = (fallback_dir / path).resolve()
        if fallback.exists():
            return fallback

    return primary


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_output_overrides(config: dict, out_dir: Path) -> dict:
    # Override evaluation output paths to keep each run isolated.
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = config.setdefault('evaluation', {})
    eval_cfg['out_area_csv'] = str(out_dir / 'ppm_area.csv')
    eval_cfg['out_plot'] = str(out_dir / 'ppm_area_plot.png')
    eval_cfg['out_sensor_csv'] = str(out_dir / 'ppm_sensor.csv')
    eval_cfg['out_prox_csv'] = str(out_dir / 'ppm_prox.csv')
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PPM with random sensor positions.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory to override evaluation outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    config = load_config(str(config_path))
    print(f"[INFO] Config loaded: {config_path}")

    if args.out_dir is not None:
        config = apply_output_overrides(config, Path(args.out_dir))
        print(f"[INFO] Output directory overridden: {args.out_dir}")

    world_file = resolve_path(
        config['io']['world_file'],
        base_dir=config_path.parent,
        fallback_dir=repo_root,
    )
    loader = WorldLoader(world_file=str(world_file))
    obstacles = loader.load_obstacles()
    print(f"[INFO] World loaded from {world_file} with {len(obstacles)} obstacles.")

    # Compute sampling bounds from the inner faces of wall_ obstacles.
    sampling_min, sampling_max = compute_inner_bounds_from_walls(obstacles, padding=0.0)
    print(f"[INFO] Sampling bounds: x[{sampling_min[0]:.2f}, {sampling_max[0]:.2f}], "
          f"y[{sampling_min[1]:.2f}, {sampling_max[1]:.2f}]")

    if config['sensor'].get('position_randomize', False):
        from modules.sensor_random import sample_sensor_positions
        print(f"[INFO] Randomizing sensor positions with method='{config['sensor'].get('sampling_method', 'random')}'.")
        expected_samples = int(config['sensor'].get('num_sampling', 10))
        sensor_positions = sample_sensor_positions(
            obstacles=obstacles,
            num_samples=expected_samples,
            x_range=(sampling_min[0], sampling_max[0]),
            y_range=(sampling_min[1], sampling_max[1]),
            seed=int(config['sensor'].get('random_seed', 0)),
            max_range_m=float(config['sensor']['max_range_m']),
            fov_deg=float(config['sensor']['angle_deg']),
            sampling_method=str(config['sensor'].get('sampling_method', 'random')),
            halton_scramble=bool(config['sensor'].get('halton_scramble', True)),
            angle_step_deg=2.0,
        )
        if len(sensor_positions) != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} sampled sensors, but got {len(sensor_positions)}."
            )
        print(f"[INFO] Sampled {len(sensor_positions)} sensor positions.")
    else:
        raise ValueError("sensor.position_randomize is false; fixed positions are not configured.")

    det = ProxDetector(obstacles=obstacles)
    max_range = float(config['sensor']['max_range_m'])
    fov_deg = float(config['sensor']['angle_deg'])
    sensor_prox: list[SensorProx] = []
    for count, sensor in enumerate(sensor_positions):
        prox = det.detect(
            sensor_xyyaw=sensor,
            max_range_m=max_range,
            fov_deg=fov_deg,
            angle_step_deg=2.0,
        )
        sensor_prox.append(SensorProx(count=count, sensor_xyyaw=sensor, prox=prox))
        if (count + 1) % 10 == 0 or (count + 1) == len(sensor_positions):
            print(f"[INFO] Detected proximity points from sensor {count+1}/{len(sensor_positions)}.")

    sensors = build_sensor_poses(sensor_positions)
    bounds_min, bounds_max = compute_bounds_from_obstacles(obstacles, padding=1.0)
    eval_cfg = config.get('evaluation', {})
    true_free_area = compute_true_free_area(
        obstacles,
        sampling_min,
        sampling_max,
    )

    log_output_numbers = eval_cfg.get('log_output_numbers', [])
    if isinstance(log_output_numbers, int):
        log_output_numbers = [log_output_numbers]

    area_records = evaluate_ppm_area_records(
        sensors=sensors,
        sensor_prox=sensor_prox,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        true_free_area=true_free_area,
    )

    csv_path = eval_cfg.get('out_area_csv', 'random_ppm_generator/output/ppm_area.csv')
    plot_path = eval_cfg.get('out_plot', 'random_ppm_generator/output/ppm_area_plot.png')

    csv_path = str(resolve_path(csv_path, base_dir=config_path.parent, fallback_dir=repo_root))
    plot_path = str(resolve_path(plot_path, base_dir=config_path.parent, fallback_dir=repo_root))
    eval_cfg['out_sensor_csv'] = str(resolve_path(
        eval_cfg.get('out_sensor_csv', 'map_builder/random_ppm_generator/output/ppm_sensor.csv'),
        base_dir=config_path.parent,
        fallback_dir=repo_root,
    ))
    eval_cfg['out_prox_csv'] = str(resolve_path(
        eval_cfg.get('out_prox_csv', 'map_builder/random_ppm_generator/output/ppm_prox.csv'),
        base_dir=config_path.parent,
        fallback_dir=repo_root,
    ))
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    write_area_csv(area_records, csv_path)
    plot_area_records(area_records, plot_path)
    save_ppm(sensor_prox=sensor_prox, config=config)

    if log_output_numbers:
        # Save proximity+PPM overlays at requested sampling counts.
        plot_path_obj = Path(plot_path)
        max_count = len(sensors)
        for k in sorted({int(n) for n in log_output_numbers if 1 <= int(n) <= max_count}):
            prox_points_k = build_prox_pointset(sensor_prox, upto_count=k)
            ppm_polygons_k = build_ppm_polygons(
                sensors=sensors[:k],
                prox_points=prox_points_k,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )
            log_plot_path = plot_path_obj.with_name(f"{plot_path_obj.stem}_{k}{plot_path_obj.suffix}")
            plot_proximity_with_ppm(
                obstacles=obstacles,
                sensor_prox=sensor_prox[:k],
                ppm_polygons=ppm_polygons_k,
                show=False,
                output_path=str(log_plot_path),
            )
            print(f"[INFO] Saved PPM plot with {k} sensors to {log_plot_path}.")

    # Always save plot at the final sampling state so runs have a definitive end snapshot.
    final_plot_path = Path(plot_path).with_name(f"{Path(plot_path).stem}_final{Path(plot_path).suffix}")
    full_prox_points = build_prox_pointset(sensor_prox, upto_count=len(sensors))
    full_ppm_polygons = build_ppm_polygons(
        sensors=sensors,
        prox_points=full_prox_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    final_plot_formats = ["png", "eps", "svg"]
    plot_proximity_with_ppm(
        obstacles=obstacles,
        sensor_prox=sensor_prox,
        ppm_polygons=full_ppm_polygons,
        show=False,
        output_path=str(final_plot_path),
        formats=final_plot_formats,
    )
    print(
        f"[INFO] Saved final PPM plot with {len(sensors)} sensors to "
        f"{final_plot_path.with_suffix('')} in formats: {', '.join(final_plot_formats)}."
    )


if __name__ == "__main__":
    main()

