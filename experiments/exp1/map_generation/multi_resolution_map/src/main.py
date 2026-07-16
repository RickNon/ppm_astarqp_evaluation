import argparse
import json
from pathlib import Path

from modules.config_loader import load_config
from modules.coverage_snapshot import CoverageThresholdSnapshotRecorder
from modules.sdf_world_parser import parse_world
from modules.obstacles import models_to_obstacles
from modules.map_domain import (
    compute_inner_domain_from_walls,
    compute_outer_domain_from_walls,
)
from modules.occupancy_classifier import OccupancyClassifier
from modules.quadtree_builder import QuadtreeBuilder, QuadtreeBuildConfig
from modules.grid_exporter import export_leaf_cells_csv
from modules.uniform_grid import (
    reconstruct_uniform_grid_from_leaves,
    export_uniform_grid_csv,
)
from modules.nav2_map_exporter import export_nav2_map_from_leaves
from modules.analysis_recorder import ResolutionAnalysisRecorder
from modules.true_free_area import compute_true_free_area
from modules.visualizer import QuadtreeVisualizer
from modules.logger import export_log_txt


def _safe_resolution_name(value: float) -> str:
    return str(value).replace(".", "p")


def _coverage_percent(target: float) -> int:
    return int(round(target * 100.0))


def main():
    # load config and parse world
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--nav2-output-dir",
        type=Path,
        default=None,
        help="Override the configured environment-level Nav2 map output directory.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.nav2_output_dir is not None:
        config.uniform_grid.nav2.output_dir = args.nav2_output_dir.resolve()
    models = parse_world(config.world_file, config)
    # compute domain and obstacles
    inner_domain = compute_inner_domain_from_walls(models)
    outer_domain = compute_outer_domain_from_walls(models)
    obstacles = models_to_obstacles(models)
    true_free_area = compute_true_free_area(obstacles, inner_domain)

    # build quadtree
    classifier = OccupancyClassifier(obstacles)

    build_config = QuadtreeBuildConfig(
        min_resolution=config.quadtree.min_resolution,
        max_depth=config.quadtree.max_depth,
        treat_mixed_as_occupied_at_min_resolution=(
            config.quadtree.treat_mixed_as_occupied_at_min_resolution
        ),
    )

    builder = QuadtreeBuilder(classifier, build_config)
    snapshot_recorder = None
    if config.uniform_grid.enable and config.uniform_grid.coverage_targets:
        if config.uniform_grid.nav2.output_dir is None:
            raise ValueError(
                "uniform_grid.nav2.output_dir is required when coverage_targets are configured"
            )
        snapshot_recorder = CoverageThresholdSnapshotRecorder(
            true_free_area=true_free_area,
            targets=config.uniform_grid.coverage_targets,
        )

    root = builder.build(inner_domain, recorder=snapshot_recorder)
    leaves = builder.collect_leaves(root)

    # summarize results
    num_free = sum(1 for n in leaves if n.state.name == "FREE")
    num_occupied = sum(1 for n in leaves if n.state.name == "OCCUPIED")
    num_mixed = sum(1 for n in leaves if n.state.name == "MIXED")

    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_models": len(models),
        "num_obstacles": len(obstacles),
        "inner_domain": {
            "xmin": inner_domain.xmin,
            "ymin": inner_domain.ymin,
            "xmax": inner_domain.xmax,
            "ymax": inner_domain.ymax,
            "width": inner_domain.width,
            "height": inner_domain.height,
        },
        "outer_domain": {
            "xmin": outer_domain.xmin,
            "ymin": outer_domain.ymin,
            "xmax": outer_domain.xmax,
            "ymax": outer_domain.ymax,
            "width": outer_domain.width,
            "height": outer_domain.height,
        },
        "quadtree": {
            "leaf_count": len(leaves),
            "free_leaf_count": num_free,
            "occupied_leaf_count": num_occupied,
            "mixed_leaf_count": num_mixed,
            "min_resolution": build_config.min_resolution,
            "max_depth": build_config.max_depth,
        },
    }

    if config.output.export_summary_json:
        # Save the build summary for reproducibility.
        with open(config.output.output_dir / "quadtree_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if config.output.export_leaf_csv:
        # Export leaf cells
        export_leaf_cells_csv(
            leaves,
            config.output.output_dir / "leaf_cells.csv",
        )

    coverage_log_lines: list[str] = []
    if snapshot_recorder is not None:
        if not snapshot_recorder.is_complete:
            missing = ", ".join(f"{target:.6g}" for target in snapshot_recorder.missing_targets)
            raise ValueError(f"The quadtree did not reach coverage targets: {missing}")

        nav2_output_dir = config.uniform_grid.nav2.output_dir
        if nav2_output_dir is None:
            raise RuntimeError("The validated Nav2 output directory is unexpectedly missing")

        environment_name = config.world_file.stem
        snapshots = snapshot_recorder.snapshots
        percentages = [_coverage_percent(target) for target in snapshot_recorder.targets]
        if len(set(percentages)) != len(percentages):
            raise ValueError("Coverage targets must map to distinct integer percentage labels")

        for target in snapshot_recorder.targets:
            snapshot = snapshots[target]
            percent = _coverage_percent(target)
            target_dir = nav2_output_dir / f"coverage_{percent:02d}"
            stem = target_dir / (
                f"{environment_name}_coverage_{percent:02d}_"
                f"r_{_safe_resolution_name(config.uniform_grid.resolution)}"
            )
            export_nav2_map_from_leaves(
                leaves=snapshot.leaves,
                inner_domain=inner_domain,
                outer_domain=outer_domain,
                models=models,
                resolution=config.uniform_grid.resolution,
                output_stem=stem,
                treat_mixed_as_occupied=config.uniform_grid.treat_mixed_as_occupied,
                occupied_thresh=config.uniform_grid.nav2.occupied_thresh,
                free_thresh=config.uniform_grid.nav2.free_thresh,
                negate=config.uniform_grid.nav2.negate,
                mode=config.uniform_grid.nav2.mode,
            )
            coverage_log_lines.append(
                f"Coverage {target:.2f}: reached={snapshot.reached_ratio:.6f}, "
                f"leaves={snapshot.leaf_count}, output={stem}"
            )

    grid = None
    grid_width = None
    grid_height = None
    if config.uniform_grid.enable and (
        config.uniform_grid.export_csv or config.uniform_grid.export_visualization
    ):
        # Reconstruct a dense grid from quadtree leaves.
        grid, grid_width, grid_height = reconstruct_uniform_grid_from_leaves(
            leaves=leaves,
            domain=inner_domain,
            resolution=config.uniform_grid.resolution,
        )

        if config.uniform_grid.export_csv:
            export_uniform_grid_csv(
                grid,
                config.output.output_dir / "uniform_grid.csv",
            )

    if config.analysis.resolution_sweep.enable:
        for min_res in config.analysis.resolution_sweep.min_resolutions:
            sweep_build_config = QuadtreeBuildConfig(
                min_resolution=min_res,
                max_depth=config.quadtree.max_depth,
                treat_mixed_as_occupied_at_min_resolution=(
                    config.quadtree.treat_mixed_as_occupied_at_min_resolution
                ),
            )

            sweep_builder = QuadtreeBuilder(classifier, sweep_build_config)
            recorder = ResolutionAnalysisRecorder(true_free_area=true_free_area)
            sweep_root = sweep_builder.build(inner_domain, recorder=recorder)
            sweep_leaves = sweep_builder.collect_leaves(sweep_root)

            # Build a quadtree while recording free-area growth after each depth-wide expansion.

            safe_name = _safe_resolution_name(min_res)

            if config.analysis.resolution_sweep.export_csv:
                recorder.export_csv(
                    config.output.output_dir / f"free_area_vs_resolution_{safe_name}.csv"
                )

            if config.analysis.resolution_sweep.export_plot:
                for fmt in config.visualization.formats:
                    recorder.export_plot(
                        config.output.output_dir / f"free_area_vs_resolution_{safe_name}.{fmt}",
                        figure_size_inch=config.visualization.figure_size_inch,
                        dpi=config.visualization.dpi,
                    )

            analysis_viz = QuadtreeVisualizer(
                outer_domain=outer_domain,
                figure_size_inch=config.visualization.figure_size_inch,
                dpi=config.visualization.dpi,
            )
            analysis_viz.draw_walls(
                models=models,
                wall_color=config.visualization.wall_color,
                line_width=config.visualization.line_width,
            )
            analysis_viz.draw_leaves(
                leaves=sweep_leaves,
                occupied_color="black",
                free_color="green",
                mixed_color="gray",
                boundary_color=config.visualization.boundary_color,
                line_width=config.visualization.line_width,
            )
            analysis_viz.finalize(show_axes=config.visualization.show_axes)
            analysis_viz.save(
                config.output.output_dir / f"quadtree_analysis_resolution_{safe_name}.png"
            )

    if config.visualization.enable:
        # Visualize quadtree leaves.
        viz = QuadtreeVisualizer(
            outer_domain=outer_domain,
            figure_size_inch=config.visualization.figure_size_inch,
            dpi=config.visualization.dpi,
        )

        viz.draw_walls(
            models=models,
            wall_color=config.visualization.wall_color,
            line_width=config.visualization.line_width,
        )

        viz.draw_leaves(
            leaves=leaves,
            occupied_color=config.visualization.occupied_color,
            free_color=config.visualization.free_color,
            mixed_color=config.visualization.boundary_color,
            boundary_color=config.visualization.boundary_color,
            line_width=config.visualization.line_width,
        )

        if config.visualization.draw_obstacles:
            viz.draw_obstacles(
                obstacles=obstacles,
                boundary_color=config.visualization.boundary_color,
                line_width=config.visualization.line_width,
            )

        viz.finalize(show_axes=config.visualization.show_axes)

        for fmt in config.visualization.formats:
            # Export quadtree visualization.
            viz.save(config.output.output_dir / f"quadtree_map.{fmt}")

    if (
        config.visualization.enable
        and config.uniform_grid.export_visualization
        and grid is not None
    ):
        # Visualize the reconstructed dense grid separately.
        viz_grid = QuadtreeVisualizer(
            outer_domain=outer_domain,
            figure_size_inch=config.visualization.figure_size_inch,
            dpi=config.visualization.dpi,
        )

        viz_grid.draw_walls(
            models=models,
            wall_color=config.visualization.wall_color,
            line_width=config.visualization.line_width,
        )

        viz_grid.draw_uniform_grid(
            grid=grid,
            domain=inner_domain,
            resolution=config.uniform_grid.resolution,
            occupied_color=config.visualization.occupied_color,
            free_color=config.visualization.free_color,
            boundary_color=config.visualization.boundary_color,
            line_width=config.visualization.line_width,
        )

        viz_grid.finalize(show_axes=config.visualization.show_axes)

        for fmt in config.visualization.formats:
            # Export reconstructed dense-grid visualization.
            viz_grid.save(config.output.output_dir / f"uniform_grid_map.{fmt}")

    log_lines = [
        f"Parsed models: {len(models)}",
        f"Obstacle count: {len(obstacles)}",
        (
            "Inner domain: "
            f"x=[{inner_domain.xmin:.3f}, {inner_domain.xmax:.3f}], "
            f"y=[{inner_domain.ymin:.3f}, {inner_domain.ymax:.3f}]"
        ),
        (
            "Outer domain: "
            f"x=[{outer_domain.xmin:.3f}, {outer_domain.xmax:.3f}], "
            f"y=[{outer_domain.ymin:.3f}, {outer_domain.ymax:.3f}]"
        ),
        f"Leaf count: {len(leaves)}",
        f"Free leaves: {num_free}",
        f"Occupied leaves: {num_occupied}",
        f"Mixed leaves: {num_mixed}",
    ]

    if grid is not None:
        log_lines.append(f"Uniform grid resolution: {config.uniform_grid.resolution}")
        log_lines.append(f"Uniform grid size: {grid_width} x {grid_height}")

    log_lines.extend(coverage_log_lines)

    for line in log_lines:
        print(line)

    if config.output.export_log_txt:
        # Save console-equivalent log lines to the output directory.
        export_log_txt(config.output.output_dir / "log.txt", log_lines)


if __name__ == "__main__":
    main()
