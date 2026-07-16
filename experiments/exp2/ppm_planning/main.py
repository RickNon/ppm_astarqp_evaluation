import argparse
from pathlib import Path

from modules.config_loader import load_config
from modules.csv_io import (
    clear_matching_files,
    read_start_goal_pairs,
    write_debug_results,
    write_path_csv,
    write_results,
    write_start_goal_pairs,
)
from modules.graph_manager import GraphManager
from modules.metrics import compute_clearance_metrics, compute_curvature_metrics
from modules.ompl_bitstar_planner import OmplBitstarPPMPlanner
from modules.planner import AstarQPBase, AstarQPFull, AstarQPWideSpace
from modules.plotter import save_trial_plot
from modules.ppm_validity import PPMValidityChecker
from modules.sampler import StartGoalSampler
from modules.world_loader import WorldLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPM path-planning experiments.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment-specific planner config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    obstacles = []
    if config.world_file is not None and config.world_file.is_file():
        obstacles = WorldLoader(config.world_file).load_obstacles()
    elif config.world_file is not None:
        print(f"World file not found; clearance metrics will be skipped: {config.world_file}")

    if config.start_goal_pairs_csv is not None:
        # Reuse exact historical start-goal pairs for comparable reruns.
        pairs = read_start_goal_pairs(config.start_goal_pairs_csv)
        print(f"Loaded {len(pairs)} start-goal pairs from: {config.start_goal_pairs_csv}")
    else:
        sampler = StartGoalSampler(
            ppm_sensor_csv=config.ppm_sensor_csv,
            obstacles=obstacles,
            random_seed=config.random_seed,
            min_start_goal_distance=config.min_start_goal_distance,
            max_sampling_attempts=config.max_sampling_attempts,
            bounds_margin_m=config.bounds_margin_m,
        )
        pairs = sampler.sample_pairs(config.num_trials)

    output_path = output_dir / "start_goal_pairs.csv"
    write_start_goal_pairs(output_path, pairs)
    print(f"Saved {len(pairs)} start-goal pairs to: {output_path}")

    graph_manager = GraphManager()
    graph_manager.load_omnia_files(str(config.ppm_sensor_csv), str(config.ppm_prox_csv))
    free_area_polygons = graph_manager.build_free_area_polygons()
    paths_by_method: dict[str, dict[int, object]] = {}
    for method_name in config.planner_methods:
        if method_name == "astar_qp_base":
            planner = AstarQPBase(
                graph_manager=graph_manager,
                qp_solver=config.qp_solver,
            )
        elif method_name == "astar_qp_wide_space":
            planner = AstarQPWideSpace(
                graph_manager=graph_manager,
                qp_solver=config.qp_solver,
            )
        elif method_name == "astar_qp_full":
            planner = AstarQPFull(
                graph_manager=graph_manager,
                qp_solver=config.qp_solver,
                mu=config.smoothing_mu,
            )
        elif method_name == "bitstar_ppm":
            ppm_validity = PPMValidityChecker(graph_manager=graph_manager)
            ppm_validity.build()
            planner = OmplBitstarPPMPlanner(
                graph_manager=graph_manager,
                validity_checker=ppm_validity,
                time_limit_s=config.bitstar_time_limit_s,
                range_m=config.bitstar_range_m,
                random_seed=config.bitstar_random_seed,
                simplify_solution=config.bitstar_simplify_solution,
                state_validity_resolution=config.bitstar_state_validity_resolution,
            )
        else:
            raise RuntimeError(f"Unsupported planner method: {method_name}")

        paths_dir = output_dir / method_name / "paths"
        clear_matching_files(paths_dir, "*_path.csv")
        paths_by_method[method_name] = {}

        result_rows = []
        debug_rows = []
        for pair in pairs:
            start = (float(pair["start_x"]), float(pair["start_y"]))
            goal = (float(pair["goal_x"]), float(pair["goal_y"]))
            planning_result = planner.plan(start=start, goal=goal)
            paths_by_method[method_name][int(pair["count"])] = (
                planning_result.path_xy if planning_result.success else None
            )
            if method_name == "bitstar_ppm":
                debug_rows.append(
                    {
                        "count": pair["count"],
                        "success": planning_result.success,
                        "error": planning_result.error or "",
                    }
                )
            result_rows.append(
                {
                    "count": pair["count"],
                    "start_x": pair["start_x"],
                    "start_y": pair["start_y"],
                    "goal_x": pair["goal_x"],
                    "goal_y": pair["goal_y"],
                    "success": planning_result.success,
                    "time": planning_result.time_ms,
                    "path_length": planning_result.path_length if planning_result.success else "",
                    "mean_abs_curvature": "",
                    "curvature_energy": "",
                    "mean_clearance": "",
                    "low_clearance_ratio": "",
                }
            )
            if planning_result.success and planning_result.path_xy is not None:
                mean_abs_curvature, curvature_energy = compute_curvature_metrics(
                    planning_result.path_xy,
                    spacing=config.resample_spacing_m,
                )
                result_rows[-1]["mean_abs_curvature"] = mean_abs_curvature
                result_rows[-1]["curvature_energy"] = curvature_energy
                if obstacles:
                    mean_clearance, low_clearance_ratio = compute_clearance_metrics(
                        planning_result.path_xy,
                        obstacles=obstacles,
                        spacing=config.resample_spacing_m,
                        low_clearance_threshold_m=config.low_clearance_threshold_m,
                    )
                    result_rows[-1]["mean_clearance"] = mean_clearance
                    result_rows[-1]["low_clearance_ratio"] = low_clearance_ratio
                path_output_path = output_dir / method_name / "paths" / f"{pair['count']}_path.csv"
                write_path_csv(
                    output_path=path_output_path,
                    count=int(pair["count"]),
                    path_xy=planning_result.path_xy,
                )

        results_path = output_dir / method_name / "results.csv"
        write_results(results_path, result_rows)
        print(f"Saved {len(result_rows)} planning results to: {results_path}")
        if method_name == "bitstar_ppm":
            debug_path = output_dir / method_name / "debug.csv"
            write_debug_results(debug_path, debug_rows)
            print(f"Saved {len(debug_rows)} debug rows to: {debug_path}")

    if config.plot_enable:
        clear_matching_files(config.plot_output_dir, "*_plot.png")
        for pair in pairs:
            count = int(pair["count"])
            start = (float(pair["start_x"]), float(pair["start_y"]))
            goal = (float(pair["goal_x"]), float(pair["goal_y"]))
            method_paths = {
                method_name: paths_by_method[method_name].get(count)
                for method_name in paths_by_method
            }
            plot_output_path = config.plot_output_dir / f"{count}_plot.png"
            save_trial_plot(
                output_path=plot_output_path,
                obstacles=obstacles,
                free_area_polygons=free_area_polygons,
                start=start,
                goal=goal,
                method_paths=method_paths,
                title=f"Trial {count}",
            )


if __name__ == "__main__":
    main()

