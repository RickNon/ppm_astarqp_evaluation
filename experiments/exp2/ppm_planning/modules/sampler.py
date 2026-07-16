from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np
import pandas as pd

from modules.world_loader import Obstacle


@dataclass(frozen=True)
class Bounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


class StartGoalSampler:
    def __init__(
        self,
        ppm_sensor_csv: Path,
        obstacles: list[Obstacle],
        random_seed: int,
        min_start_goal_distance: float,
        max_sampling_attempts: int,
        bounds_margin_m: float,
    ) -> None:
        self.ppm_sensor_csv = ppm_sensor_csv
        self.obstacles = obstacles
        self.rng = np.random.default_rng(random_seed)
        self.min_start_goal_distance = min_start_goal_distance
        self.max_sampling_attempts = max_sampling_attempts
        self.bounds_margin_m = bounds_margin_m
        self.bounds = self._load_bounds_from_sensor_csv()

    def _load_bounds_from_sensor_csv(self) -> Bounds:
        frame = pd.read_csv(self.ppm_sensor_csv)
        x_values = frame["location_odometry_pos_x"].astype(float).to_numpy()
        y_values = frame["location_odometry_pos_y"].astype(float).to_numpy()

        return Bounds(
            min_x=float(np.min(x_values) - self.bounds_margin_m),
            max_x=float(np.max(x_values) + self.bounds_margin_m),
            min_y=float(np.min(y_values) - self.bounds_margin_m),
            max_y=float(np.max(y_values) + self.bounds_margin_m),
        )

    def _sample_point(self) -> tuple[float, float]:
        x = float(self.rng.uniform(self.bounds.min_x, self.bounds.max_x))
        y = float(self.rng.uniform(self.bounds.min_y, self.bounds.max_y))
        return x, y

    def _is_point_valid(self, point: tuple[float, float]) -> bool:
        px, py = point
        for obstacle in self.obstacles:
            if obstacle.contains_point(px, py):
                return False
        return True

    def _sample_valid_point(self) -> tuple[float, float]:
        for _ in range(self.max_sampling_attempts):
            point = self._sample_point()
            if self._is_point_valid(point):
                return point
        raise RuntimeError("Failed to sample a valid point outside obstacles within the attempt limit.")

    def sample_pairs(self, num_trials: int) -> list[dict[str, float | int]]:
        pairs: list[dict[str, float | int]] = []
        for count in range(1, num_trials + 1):
            for _ in range(self.max_sampling_attempts):
                start = self._sample_valid_point()
                goal = self._sample_valid_point()
                if hypot(goal[0] - start[0], goal[1] - start[1]) < self.min_start_goal_distance:
                    continue
                pairs.append(
                    {
                        "count": count,
                        "start_x": start[0],
                        "start_y": start[1],
                        "goal_x": goal[0],
                        "goal_y": goal[1],
                    }
                )
                break
            else:
                raise RuntimeError(
                    f"Failed to sample a valid start-goal pair for count={count} within the attempt limit."
                )
        return pairs

