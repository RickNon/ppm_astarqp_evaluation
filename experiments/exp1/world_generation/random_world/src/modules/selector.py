from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from modules.config_loader import PlacementConfig, WorldConfig
from modules.sampler import SamplingResult


@dataclass
class SelectionResult:
    selected_xy: np.ndarray
    selected_density: np.ndarray


class ObstacleCenterSelector:
    """Select final obstacle centers under min-spacing and boundary constraints."""

    def __init__(self, world_cfg: WorldConfig, placement_cfg: PlacementConfig) -> None:
        self.world_cfg = world_cfg
        self.placement_cfg = placement_cfg

    def select(self, sampling_result: SamplingResult) -> SelectionResult:
        inside_mask = self._apply_boundary_margin(sampling_result.accepted_xy)
        filtered_xy = sampling_result.accepted_xy[inside_mask]
        filtered_density = sampling_result.accepted_density[inside_mask]

        ordered_indices = np.lexsort((filtered_xy[:, 1], filtered_xy[:, 0], -filtered_density))

        selected_points: List[np.ndarray] = []
        selected_density: List[float] = []
        spatial_hash: Dict[Tuple[int, int], List[int]] = {}
        cell_size = self.placement_cfg.min_spacing

        for idx in ordered_indices:
            point = filtered_xy[idx]
            if self._is_far_enough(point, selected_points, spatial_hash, cell_size):
                selected_index = len(selected_points)
                selected_points.append(point)
                selected_density.append(float(filtered_density[idx]))
                cell = self._cell_index(point, cell_size)
                spatial_hash.setdefault(cell, []).append(selected_index)

                if len(selected_points) >= self.placement_cfg.max_obstacles:
                    break

        if not selected_points:
            return SelectionResult(np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64))

        return SelectionResult(np.vstack(selected_points), np.asarray(selected_density, dtype=np.float64))

    def _apply_boundary_margin(self, xy: np.ndarray) -> np.ndarray:
        m = self.placement_cfg.boundary_margin
        return (
            (xy[:, 0] >= m)
            & (xy[:, 0] <= self.world_cfg.width - m)
            & (xy[:, 1] >= m)
            & (xy[:, 1] <= self.world_cfg.height - m)
        )

    def _is_far_enough(
        self,
        point: np.ndarray,
        selected_points: List[np.ndarray],
        spatial_hash: Dict[Tuple[int, int], List[int]],
        cell_size: float,
    ) -> bool:
        if not selected_points:
            return True

        px, py = self._cell_index(point, cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for selected_idx in spatial_hash.get((px + dx, py + dy), []):
                    if np.linalg.norm(point - selected_points[selected_idx]) < self.placement_cfg.min_spacing:
                        return False
        return True

    @staticmethod
    def _cell_index(point: np.ndarray, cell_size: float) -> Tuple[int, int]:
        return int(point[0] // cell_size), int(point[1] // cell_size)
