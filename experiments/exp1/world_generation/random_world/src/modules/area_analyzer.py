from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from modules.config_loader import AnalysisConfig, WorldConfig
from modules.obstacle_generator import ObstacleInstance


@dataclass(frozen=True)
class AreaAnalysisResult:
    inner_area: float
    occupied_area: float
    free_area: float
    occupied_ratio: float
    free_ratio: float
    grid_shape: tuple[int, int]
    area_resolution: float


class FreeSpaceAreaAnalyzer:
    """Compute 2D occupied/free area inside the boundary walls.

    The calculation is done on a fine occupancy grid and naturally handles overlap
    because occupancy is accumulated into a single boolean mask.
    """

    def __init__(self, world_cfg: WorldConfig, analysis_cfg: AnalysisConfig) -> None:
        self.world_cfg = world_cfg
        self.analysis_cfg = analysis_cfg

    def analyze(self, obstacles: list[ObstacleInstance]) -> AreaAnalysisResult:
        res = self.analysis_cfg.area_resolution
        x_centers = np.arange(0.5 * res, self.world_cfg.width, res)
        y_centers = np.arange(0.5 * res, self.world_cfg.height, res)

        xx, yy = np.meshgrid(x_centers, y_centers)
        occupied = np.zeros_like(xx, dtype=bool)

        for obstacle in obstacles:
            occupied |= self._occupancy_mask(xx, yy, obstacle)

        cell_area = res * res
        occupied_area = float(np.count_nonzero(occupied) * cell_area)
        inner_area = float(self.world_cfg.width * self.world_cfg.height)
        free_area = max(0.0, inner_area - occupied_area)
        occupied_ratio = occupied_area / inner_area if inner_area > 0.0 else 0.0
        free_ratio = free_area / inner_area if inner_area > 0.0 else 0.0

        return AreaAnalysisResult(
            inner_area=inner_area,
            occupied_area=occupied_area,
            free_area=free_area,
            occupied_ratio=occupied_ratio,
            free_ratio=free_ratio,
            grid_shape=(len(y_centers), len(x_centers)),
            area_resolution=res,
        )

    def _occupancy_mask(self, xx: np.ndarray, yy: np.ndarray, obstacle: ObstacleInstance) -> np.ndarray:
        if obstacle.kind == "box":
            dx = xx - obstacle.center_x
            dy = yy - obstacle.center_y
            c = math.cos(-obstacle.yaw)
            s = math.sin(-obstacle.yaw)
            local_x = c * dx - s * dy
            local_y = s * dx + c * dy

            return (
                (np.abs(local_x) <= 0.5 * float(obstacle.size_x))
                & (np.abs(local_y) <= 0.5 * float(obstacle.size_y))
            )

        if obstacle.kind == "cylinder":
            dx = xx - obstacle.center_x
            dy = yy - obstacle.center_y
            return dx * dx + dy * dy <= float(obstacle.radius) ** 2

        raise ValueError(f"Unsupported obstacle kind: {obstacle.kind}")
