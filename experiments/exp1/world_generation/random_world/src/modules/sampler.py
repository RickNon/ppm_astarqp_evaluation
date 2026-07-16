from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

from modules.config_loader import SamplingConfig, WorldConfig
from modules.noise_field import DensityFieldResult


@dataclass
class SamplingResult:
    candidates_xy: np.ndarray
    accepted_xy: np.ndarray
    accepted_density: np.ndarray


class CandidateSampler:
    """Generate Sobol candidates and keep them with density-based acceptance."""

    def __init__(self, world_cfg: WorldConfig, sampling_cfg: SamplingConfig) -> None:
        self.world_cfg = world_cfg
        self.sampling_cfg = sampling_cfg

    def sample(self, density_result: DensityFieldResult) -> SamplingResult:
        candidates_xy = self._generate_candidates()
        candidate_density = self._sample_density(density_result, candidates_xy)
        accepted_probability = self._density_to_probability(candidate_density)
        accepted_mask = self._accept_candidates(accepted_probability)

        return SamplingResult(
            candidates_xy=candidates_xy,
            accepted_xy=candidates_xy[accepted_mask],
            accepted_density=candidate_density[accepted_mask],
        )

    def _generate_candidates(self) -> np.ndarray:
        if self.sampling_cfg.engine.lower() != "sobol":
            raise ValueError(f"Unsupported sampling engine: {self.sampling_cfg.engine}")

        sampler = qmc.Sobol(d=2, scramble=self.sampling_cfg.sobol_scramble, seed=self.sampling_cfg.random_seed)
        power = int(np.ceil(np.log2(max(1, self.sampling_cfg.num_candidates))))
        points = sampler.random_base2(m=power)[: self.sampling_cfg.num_candidates]
        points[:, 0] *= self.world_cfg.width
        points[:, 1] *= self.world_cfg.height
        return points

    def _sample_density(self, density_result: DensityFieldResult, xy: np.ndarray) -> np.ndarray:
        interpolator = RegularGridInterpolator(
            (density_result.y_coords, density_result.x_coords),
            density_result.density,
            bounds_error=False,
            fill_value=None,
        )
        query_points = np.column_stack((xy[:, 1], xy[:, 0]))
        return np.clip(interpolator(query_points), 0.0, 1.0)

    def _density_to_probability(self, density: np.ndarray) -> np.ndarray:
        probability = self.sampling_cfg.acceptance_bias + self.sampling_cfg.acceptance_gain * density
        return np.clip(probability, 0.0, 1.0)

    def _accept_candidates(self, probability: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.sampling_cfg.random_seed)
        draws = rng.random(len(probability))
        return draws < probability
