from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

import numpy as np

from modules.config_loader import GenerationConfig, ObstacleCatalogConfig, ObstacleTypeConfig, WorldConfig
from modules.selector import SelectionResult


@dataclass
class ObstacleInstance:
    name: str
    template_name: str
    kind: str
    center_x: float
    center_y: float
    yaw: float
    height: float
    size_x: float | None = None
    size_y: float | None = None
    radius: float | None = None


class ObstacleGenerator:
    """Assign random obstacle templates and sizes to selected centers."""

    def __init__(
        self,
        world_cfg: WorldConfig,
        generation_cfg: GenerationConfig,
        catalog_cfg: ObstacleCatalogConfig,
    ) -> None:
        self.world_cfg = world_cfg
        self.generation_cfg = generation_cfg
        self.catalog_cfg = catalog_cfg
        self.rng = np.random.default_rng(generation_cfg.random_seed)

    def generate(self, selection_result: SelectionResult) -> List[ObstacleInstance]:
        enabled_names = list(self.catalog_cfg.enabled_types)
        weights = np.asarray([self.catalog_cfg.types[name].weight for name in enabled_names], dtype=np.float64)
        weights = weights / weights.sum()

        obstacles: List[ObstacleInstance] = []
        for idx, center in enumerate(selection_result.selected_xy):
            type_name = str(self.rng.choice(enabled_names, p=weights))
            type_cfg = self.catalog_cfg.types[type_name]
            obstacle = self._instantiate_obstacle(idx, type_name, type_cfg, center)
            obstacles.append(obstacle)

        return obstacles

    def _instantiate_obstacle(
        self,
        idx: int,
        type_name: str,
        type_cfg: ObstacleTypeConfig,
        center: np.ndarray,
    ) -> ObstacleInstance:
        yaw = self._sample_yaw(type_cfg.yaw_mode)
        height = self.generation_cfg.default_obstacle_height

        if type_cfg.kind == "box":
            size_x = self._uniform_from_range(type_cfg.size_x_range)
            size_y = self._uniform_from_range(type_cfg.size_y_range)
            center_x, center_y = self._fit_box_inside_world(center[0], center[1], size_x, size_y, yaw)
            return ObstacleInstance(
                name=f"obstacle_{idx:04d}",
                template_name=type_name,
                kind="box",
                center_x=center_x,
                center_y=center_y,
                yaw=yaw,
                height=height,
                size_x=size_x,
                size_y=size_y,
            )

        if type_cfg.kind == "cylinder":
            radius = self._uniform_from_range(type_cfg.radius_range)
            center_x, center_y = self._fit_cylinder_inside_world(center[0], center[1], radius)
            return ObstacleInstance(
                name=f"obstacle_{idx:04d}",
                template_name=type_name,
                kind="cylinder",
                center_x=center_x,
                center_y=center_y,
                yaw=yaw,
                height=height,
                radius=radius,
            )

        raise ValueError(f"Unsupported obstacle kind: {type_cfg.kind}")

    def _sample_yaw(self, yaw_mode: str) -> float:
        if yaw_mode == "axis_aligned":
            return float(self.rng.choice([0.0, math.pi * 0.5]))
        if yaw_mode == "axis_aligned_or_45":
            return float(self.rng.choice([0.0, math.pi * 0.25, math.pi * 0.5, math.pi * 0.75]))
        if yaw_mode == "free":
            return float(self.rng.uniform(0.0, 2.0 * math.pi))
        raise ValueError(f"Unsupported yaw_mode: {yaw_mode}")

    def _fit_box_inside_world(self, cx: float, cy: float, sx: float, sy: float, yaw: float) -> tuple[float, float]:
        # English comment: Clamp the rotated box by its axis-aligned envelope so it stays inside the world.
        half_extent_x = 0.5 * (abs(math.cos(yaw)) * sx + abs(math.sin(yaw)) * sy)
        half_extent_y = 0.5 * (abs(math.sin(yaw)) * sx + abs(math.cos(yaw)) * sy)
        clamped_x = float(np.clip(cx, half_extent_x, self.world_cfg.width - half_extent_x))
        clamped_y = float(np.clip(cy, half_extent_y, self.world_cfg.height - half_extent_y))
        return clamped_x, clamped_y

    def _fit_cylinder_inside_world(self, cx: float, cy: float, radius: float) -> tuple[float, float]:
        clamped_x = float(np.clip(cx, radius, self.world_cfg.width - radius))
        clamped_y = float(np.clip(cy, radius, self.world_cfg.height - radius))
        return clamped_x, clamped_y

    def _uniform_from_range(self, values: tuple[float, float] | None) -> float:
        if values is None:
            raise ValueError("Required range is missing in obstacle config.")
        return float(self.rng.uniform(values[0], values[1]))
