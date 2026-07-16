from __future__ import annotations

import argparse
from pathlib import Path

from modules.config_loader import ConfigLoader
from modules.maze_geometry import MazeGeometryBuilder
from modules.maze_topology import MazeTopologyGenerator
from modules.preview_renderer import MazePreviewRenderer
from modules.sdf_writer import SdfWorldWriter


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description="Generate a maze SDF world from YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(repository_root / "configs" / "worlds" / "03_maze_wide_world_generation.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigLoader.load(args.config)

    topology = MazeTopologyGenerator(
        cols=config.maze.cols,
        rows=config.maze.rows,
        seed=config.maze.seed,
    ).generate()
    walls = MazeGeometryBuilder(config).build_walls(topology)

    output_path = Path(config.output.output_dir) / config.output.world_file
    written_path = SdfWorldWriter(config).write(walls, output_path)
    preview_path = Path(config.visualization.output_dir) / config.visualization.preview_image
    rendered_preview_path = MazePreviewRenderer(config).render(walls, preview_path)
    print(f"[OK] Generated maze world: {written_path}")
    print(f"[OK] Generated preview image: {rendered_preview_path}")
    print(f"[INFO] corridor_width={config.layout.corridor_width:.6f} m")
    print(f"[INFO] wall_count={len(walls)}")


if __name__ == "__main__":
    main()
