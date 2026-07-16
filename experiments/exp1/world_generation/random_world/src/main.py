from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from modules.area_analyzer import FreeSpaceAreaAnalyzer
from modules.config_loader import ConfigLoader, WorldGenerationConfig
from modules.logger import LogWriter
from modules.noise_field import DensityFieldGenerator
from modules.obstacle_generator import ObstacleGenerator
from modules.sampler import CandidateSampler
from modules.sdf_writer import SdfWorldWriter
from modules.selector import ObstacleCenterSelector
from modules.visualizer import LayoutVisualizer, Visualizer


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[5]
    default_config = repository_root / "configs" / "worlds" / "01_sparse_world_generation.yaml"
    parser = argparse.ArgumentParser(
        description="Unified world generator: density field, candidate sampling, center selection, and SDF export."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to the unified YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for the output directory defined in the YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader.load(args.config)

    output_dir = Path(args.output_dir or cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density_result = DensityFieldGenerator(cfg.world, cfg.grid, cfg.noise).generate()
    sampling_result = CandidateSampler(cfg.world, cfg.sampling).sample(density_result)
    selection_result = ObstacleCenterSelector(cfg.world, cfg.placement).select(sampling_result)
    obstacles = ObstacleGenerator(cfg.world, cfg.generation, cfg.obstacle_catalog).generate(selection_result)
    area_result = FreeSpaceAreaAnalyzer(cfg.world, cfg.analysis).analyze(obstacles)

    diagnostic_visualizer = Visualizer(cfg.world, cfg.visualization, cfg.placement)
    _save_pipeline_images(cfg, diagnostic_visualizer, density_result, sampling_result, selection_result, obstacles, output_dir)

    sdf_path = output_dir / f"{cfg.world.name}.world"
    SdfWorldWriter(cfg.world, cfg.generation).write(obstacles, sdf_path)
    LogWriter().write(output_dir / "log.txt", cfg.world.name, obstacles, area_result)

    type_counter = Counter(obstacle.template_name for obstacle in obstacles)

    print("Unified generation completed.")
    print(f"Accepted candidates: {sampling_result.accepted_xy.shape[0]}")
    print(f"Selected centers: {selection_result.selected_xy.shape[0]}")
    print(f"Generated obstacles: {len(obstacles)}")
    print(f"World file: {sdf_path}")
    print(f"Free area [m^2]: {area_result.free_area:.6f}")
    print(f"Occupied area [m^2]: {area_result.occupied_area:.6f}")
    print("Obstacle counts by template:")
    for name, count in sorted(type_counter.items()):
        print(f"  {name}: {count}")


def _save_pipeline_images(
    cfg: WorldGenerationConfig,
    visualizer: Visualizer,
    density_result,
    sampling_result,
    selection_result,
    obstacles,
    output_dir: Path,
) -> None:
    if cfg.visualization.save_density_only:
        visualizer.save_density_only(
            density_result,
            output_dir / _numbered_image_name(1, cfg.visualization.density_image_name),
        )
    if cfg.visualization.save_candidates_overlay:
        visualizer.save_candidates_overlay(
            density_result,
            sampling_result,
            output_dir / _numbered_image_name(2, "sobol_candidates_overlay.png"),
        )
    if cfg.visualization.save_accepted_overlay:
        visualizer.save_accepted_overlay(
            density_result,
            sampling_result,
            output_dir / _numbered_image_name(3, "accepted_candidates_overlay.png"),
        )
    if cfg.visualization.save_selected_overlay:
        visualizer.save_selected_overlay(
            density_result,
            selection_result,
            output_dir / _numbered_image_name(4, "selected_obstacle_centers_overlay.png"),
        )
    if cfg.visualization.save_layout_preview:
        LayoutVisualizer(cfg.world, cfg.visualization).save(
            obstacles,
            output_dir / _numbered_image_name(5, "layout_preview.png"),
        )


def _numbered_image_name(order: int, base_name: str) -> str:
    return f"{order:02d}_{base_name}"


if __name__ == "__main__":
    main()
