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
class GridConfig:
    resolution: float


@dataclass(frozen=True)
class NoiseConfig:
    seed: int
    scale: float
    octaves: int
    persistence: float
    lacunarity: float
    normalize_min: float
    normalize_max: float


@dataclass(frozen=True)
class SamplingConfig:
    engine: str
    num_candidates: int
    sobol_scramble: bool
    acceptance_bias: float
    acceptance_gain: float
    random_seed: int


@dataclass(frozen=True)
class PlacementConfig:
    min_spacing: float
    selection_policy: str
    max_obstacles: int
    boundary_margin: float


@dataclass(frozen=True)
class GenerationConfig:
    random_seed: int
    z_base: float
    default_obstacle_height: float


@dataclass(frozen=True)
class AnalysisConfig:
    area_resolution: float


@dataclass(frozen=True)
class VisualizationConfig:
    dpi: int
    output_dir: str
    density_image_name: str
    show_axes: bool
    point_size: float
    candidate_alpha: float
    accepted_alpha: float
    selected_alpha: float
    draw_spacing_circle: bool
    save_density_only: bool
    save_candidates_overlay: bool
    save_accepted_overlay: bool
    save_selected_overlay: bool
    obstacle_alpha: float
    wall_alpha: float
    save_layout_preview: bool


@dataclass(frozen=True)
class ObstacleTypeConfig:
    kind: str
    weight: float
    yaw_mode: str
    size_x_range: tuple[float, float] | None = None
    size_y_range: tuple[float, float] | None = None
    radius_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class ObstacleCatalogConfig:
    enabled_types: list[str]
    types: dict[str, ObstacleTypeConfig]


@dataclass(frozen=True)
class WorldGenerationConfig:
    world: WorldConfig
    grid: GridConfig
    noise: NoiseConfig
    sampling: SamplingConfig
    placement: PlacementConfig
    generation: GenerationConfig
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    obstacle_catalog: ObstacleCatalogConfig


class ConfigLoader:
    """Load YAML configuration for the unified world-generation pipeline."""

    _WORLD_DEFAULTS: dict[str, Any] = {
        "name": "generated_world",
        "wall_height": 2.0,
        "wall_thickness": 0.5,
        "ground_thickness": 0.5,
    }
    _GRID_DEFAULTS: dict[str, Any] = {
        "resolution": 0.2,
    }
    _NOISE_DEFAULTS: dict[str, Any] = {
        "normalize_min": 0.0,
        "normalize_max": 1.0,
    }
    _SAMPLING_DEFAULTS: dict[str, Any] = {
        "engine": "sobol",
        "sobol_scramble": True,
        "acceptance_bias": 0.05,
        "acceptance_gain": 0.95,
        "random_seed": 1234,
    }
    _PLACEMENT_DEFAULTS: dict[str, Any] = {
        "selection_policy": "density_descending",
        "max_obstacles": 300,
        "boundary_margin": 0.6,
    }
    _GENERATION_DEFAULTS: dict[str, Any] = {
        "random_seed": 2026,
        "z_base": 0.0,
        "default_obstacle_height": 2.0,
    }
    _ANALYSIS_DEFAULTS: dict[str, Any] = {
        "area_resolution": 0.05,
    }
    _VISUALIZATION_DEFAULTS: dict[str, Any] = {
        "dpi": 180,
        "output_dir": "output",
        "density_image_name": "density_field.png",
        "show_axes": True,
        "point_size": 8.0,
        "candidate_alpha": 0.28,
        "accepted_alpha": 0.75,
        "selected_alpha": 0.95,
        "draw_spacing_circle": False,
        "save_density_only": False,
        "save_candidates_overlay": False,
        "save_accepted_overlay": False,
        "save_selected_overlay": False,
        "obstacle_alpha": 0.80,
        "wall_alpha": 0.0,
        "save_layout_preview": False,
    }
    _OBSTACLE_CATALOG_DEFAULTS: dict[str, Any] = {
        "enabled_types": [],
        "types": {},
    }

    @staticmethod
    def load(config_path: str | Path) -> WorldGenerationConfig:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        raw_catalog = ConfigLoader._merge_section(raw, "obstacle_catalog", ConfigLoader._OBSTACLE_CATALOG_DEFAULTS)
        type_configs: dict[str, ObstacleTypeConfig] = {}
        for name, values in raw_catalog["types"].items():
            converted = dict(values)
            for key in ("size_x_range", "size_y_range", "radius_range"):
                if key in converted and converted[key] is not None:
                    converted[key] = tuple(float(v) for v in converted[key])
            type_configs[name] = ObstacleTypeConfig(**converted)

        config = WorldGenerationConfig(
            world=WorldConfig(**ConfigLoader._merge_section(raw, "world", ConfigLoader._WORLD_DEFAULTS)),
            grid=GridConfig(**ConfigLoader._merge_section(raw, "grid", ConfigLoader._GRID_DEFAULTS)),
            noise=NoiseConfig(**ConfigLoader._merge_section(raw, "noise", ConfigLoader._NOISE_DEFAULTS)),
            sampling=SamplingConfig(**ConfigLoader._merge_section(raw, "sampling", ConfigLoader._SAMPLING_DEFAULTS)),
            placement=PlacementConfig(**ConfigLoader._merge_section(raw, "placement", ConfigLoader._PLACEMENT_DEFAULTS)),
            generation=GenerationConfig(**ConfigLoader._merge_section(raw, "generation", ConfigLoader._GENERATION_DEFAULTS)),
            analysis=AnalysisConfig(**ConfigLoader._merge_section(raw, "analysis", ConfigLoader._ANALYSIS_DEFAULTS)),
            visualization=VisualizationConfig(
                **ConfigLoader._merge_section(raw, "visualization", ConfigLoader._VISUALIZATION_DEFAULTS)
            ),
            obstacle_catalog=ObstacleCatalogConfig(
                enabled_types=list(raw_catalog["enabled_types"]),
                types=type_configs,
            ),
        )
        ConfigLoader._validate(config)
        return config

    @staticmethod
    def _merge_section(raw: dict[str, Any], section_name: str, defaults: dict[str, Any]) -> dict[str, Any]:
        merged = dict(defaults)
        merged.update(raw.get(section_name, {}))
        return merged

    @staticmethod
    def _validate(config: WorldGenerationConfig) -> None:
        if config.world.width <= 0.0 or config.world.height <= 0.0:
            raise ValueError("world.width and world.height must be positive.")
        if config.grid.resolution <= 0.0:
            raise ValueError("grid.resolution must be positive.")
        if config.noise.scale <= 0.0:
            raise ValueError("noise.scale must be positive.")
        if config.sampling.num_candidates <= 0:
            raise ValueError("sampling.num_candidates must be positive.")
        if config.placement.min_spacing < config.grid.resolution:
            raise ValueError("placement.min_spacing must be >= grid.resolution.")
        if config.world.wall_thickness <= 0.0:
            raise ValueError("world.wall_thickness must be positive.")
        if config.world.wall_height <= 0.0:
            raise ValueError("world.wall_height must be positive.")
        if config.world.ground_thickness <= 0.0:
            raise ValueError("world.ground_thickness must be positive.")
        if config.generation.default_obstacle_height <= 0.0:
            raise ValueError("generation.default_obstacle_height must be positive.")
        if config.analysis.area_resolution <= 0.0:
            raise ValueError("analysis.area_resolution must be positive.")
        if config.visualization.point_size <= 0.0:
            raise ValueError("visualization.point_size must be positive.")
        if config.visualization.dpi <= 0:
            raise ValueError("visualization.dpi must be positive.")
        if not config.obstacle_catalog.enabled_types:
            raise ValueError("At least one obstacle type must be enabled.")

        for obstacle_name in config.obstacle_catalog.enabled_types:
            if obstacle_name not in config.obstacle_catalog.types:
                raise ValueError(f"Enabled obstacle type '{obstacle_name}' is not defined.")
            obstacle_cfg = config.obstacle_catalog.types[obstacle_name]
            if obstacle_cfg.weight <= 0.0:
                raise ValueError(f"Obstacle type '{obstacle_name}' must have positive weight.")

