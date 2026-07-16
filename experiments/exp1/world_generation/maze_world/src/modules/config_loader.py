from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WorldConfig:
    name: str
    width: float
    height: float
    wall_height: float
    wall_thickness: float
    ground_thickness: float


@dataclass(frozen=True)
class MazeConfig:
    cols: int
    rows: int
    corridor_width: float
    fit_mode: str
    seed: int


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str
    world_file: str


@dataclass(frozen=True)
class VisualizationConfig:
    output_dir: str
    preview_image: str
    dpi: int
    wall_alpha: float
    figure_size: float


@dataclass(frozen=True)
class MazeLayout:
    corridor_width: float
    total_width: float
    total_height: float


@dataclass(frozen=True)
class MazeWorldConfig:
    world: WorldConfig
    maze: MazeConfig
    output: OutputConfig
    visualization: VisualizationConfig
    layout: MazeLayout


class ConfigLoader:
    """Load YAML configuration for the maze-world generator."""

    _WORLD_DEFAULTS: dict[str, Any] = {
        "name": "maze_world",
        "wall_height": 2.0,
        "wall_thickness": 0.5,
        "ground_thickness": 0.5,
    }
    _MAZE_DEFAULTS: dict[str, Any] = {
        "corridor_width": 2.0,
        "fit_mode": "adjust_corridor_width",
        "seed": 42,
    }
    _OUTPUT_DEFAULTS: dict[str, Any] = {
        "output_dir": "world_maze_generator/output",
        "world_file": "maze.world",
    }
    _VISUALIZATION_DEFAULTS: dict[str, Any] = {
        "output_dir": "world_maze_generator/output",
        "preview_image": "maze_preview.png",
        "dpi": 180,
        "wall_alpha": 0.85,
        "figure_size": 7.5,
    }

    @staticmethod
    def load(config_path: str | Path) -> MazeWorldConfig:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        config = MazeWorldConfig(
            world=WorldConfig(**ConfigLoader._merge_section(raw, "world", ConfigLoader._WORLD_DEFAULTS)),
            maze=MazeConfig(**ConfigLoader._merge_section(raw, "maze", ConfigLoader._MAZE_DEFAULTS)),
            output=OutputConfig(**ConfigLoader._merge_section(raw, "output", ConfigLoader._OUTPUT_DEFAULTS)),
            visualization=VisualizationConfig(
                **ConfigLoader._merge_section(raw, "visualization", ConfigLoader._VISUALIZATION_DEFAULTS)
            ),
            layout=MazeLayout(corridor_width=0.0, total_width=0.0, total_height=0.0),
        )
        return ConfigLoader._derive_layout(config)

    @staticmethod
    def _merge_section(raw: dict[str, Any], section_name: str, defaults: dict[str, Any]) -> dict[str, Any]:
        merged = dict(defaults)
        merged.update(raw.get(section_name, {}))
        return merged

    @staticmethod
    def _derive_layout(config: MazeWorldConfig) -> MazeWorldConfig:
        ConfigLoader._validate_basics(config)

        world = config.world
        maze = config.maze
        if maze.fit_mode != "adjust_corridor_width":
            raise ValueError("maze.fit_mode must be 'adjust_corridor_width'.")

        available_width = world.width - (maze.cols - 1) * world.wall_thickness
        available_height = world.height - (maze.rows - 1) * world.wall_thickness
        if available_width <= 0.0 or available_height <= 0.0:
            raise ValueError("world dimensions are too small for the requested maze cell counts and wall thickness.")

        corridor_width_from_width = available_width / maze.cols
        corridor_width_from_height = available_height / maze.rows
        tolerance = 1e-9
        if abs(corridor_width_from_width - corridor_width_from_height) > tolerance:
            raise ValueError(
                "world dimensions are inconsistent with maze.cols/rows for a single corridor width."
            )

        corridor_width = corridor_width_from_width
        if corridor_width <= 0.0:
            raise ValueError("derived corridor width must be positive.")

        # Keep the requested value as input metadata, but always use the derived value for geometry.
        derived_maze = MazeConfig(
            cols=maze.cols,
            rows=maze.rows,
            corridor_width=corridor_width,
            fit_mode=maze.fit_mode,
            seed=maze.seed,
        )
        layout = MazeLayout(
            corridor_width=corridor_width,
            total_width=world.width,
            total_height=world.height,
        )
        return MazeWorldConfig(
            world=world,
            maze=derived_maze,
            output=config.output,
            visualization=config.visualization,
            layout=layout,
        )

    @staticmethod
    def _validate_basics(config: MazeWorldConfig) -> None:
        if config.world.width <= 0.0 or config.world.height <= 0.0:
            raise ValueError("world.width and world.height must be positive.")
        if config.world.wall_height <= 0.0:
            raise ValueError("world.wall_height must be positive.")
        if config.world.wall_thickness <= 0.0:
            raise ValueError("world.wall_thickness must be positive.")
        if config.world.ground_thickness <= 0.0:
            raise ValueError("world.ground_thickness must be positive.")
        if config.maze.cols <= 0 or config.maze.rows <= 0:
            raise ValueError("maze.cols and maze.rows must be positive.")
        if config.maze.corridor_width <= 0.0:
            raise ValueError("maze.corridor_width must be positive.")
        if config.visualization.dpi <= 0:
            raise ValueError("visualization.dpi must be positive.")
        if config.visualization.figure_size <= 0.0:
            raise ValueError("visualization.figure_size must be positive.")

