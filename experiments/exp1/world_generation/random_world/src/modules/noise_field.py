from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from modules.config_loader import NoiseConfig, WorldConfig, GridConfig


@dataclass
class DensityFieldResult:
    x_coords: np.ndarray
    y_coords: np.ndarray
    density: np.ndarray


class FractalValueNoise2D:
    """A lightweight smooth spatial field generator."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def sample(self, x: np.ndarray, y: np.ndarray, scale: float, octaves: int, persistence: float, lacunarity: float) -> np.ndarray:
        total = np.zeros_like(x, dtype=np.float64)
        amplitude = 1.0
        frequency = 1.0 / max(scale, 1e-9)
        amplitude_sum = 0.0

        for octave in range(octaves):
            total += amplitude * self._single_octave(x * frequency, y * frequency, octave)
            amplitude_sum += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        if amplitude_sum <= 0.0:
            return total
        return total / amplitude_sum

    def _single_octave(self, x: np.ndarray, y: np.ndarray, octave: int) -> np.ndarray:
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = self._fade(x - x0)
        sy = self._fade(y - y0)

        n00 = self._random_grid_value(x0, y0, octave)
        n10 = self._random_grid_value(x1, y0, octave)
        n01 = self._random_grid_value(x0, y1, octave)
        n11 = self._random_grid_value(x1, y1, octave)

        ix0 = self._lerp(n00, n10, sx)
        ix1 = self._lerp(n01, n11, sx)
        return self._lerp(ix0, ix1, sy)

    def _random_grid_value(self, x: np.ndarray, y: np.ndarray, octave: int) -> np.ndarray:
        state = (
            x * np.int64(73856093)
            ^ y * np.int64(19349663)
            ^ np.int64(self.seed + 911 * octave) * np.int64(83492791)
        ).astype(np.uint64)

        state ^= state >> np.uint64(33)
        state *= np.uint64(0xff51afd7ed558ccd)
        state ^= state >> np.uint64(33)
        state *= np.uint64(0xc4ceb9fe1a85ec53)
        state ^= state >> np.uint64(33)

        unit = (state & np.uint64((1 << 53) - 1)).astype(np.float64) / float(1 << 53)
        return 2.0 * unit - 1.0

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        return a + t * (b - a)


class DensityFieldGenerator:
    """Generate a normalized density field on a shared grid."""

    def __init__(self, world_cfg: WorldConfig, grid_cfg: GridConfig, noise_cfg: NoiseConfig) -> None:
        self.world_cfg = world_cfg
        self.grid_cfg = grid_cfg
        self.noise_cfg = noise_cfg
        self.noise = FractalValueNoise2D(seed=noise_cfg.seed)

    def generate(self) -> DensityFieldResult:
        x_coords = np.arange(0.0, self.world_cfg.width + self.grid_cfg.resolution, self.grid_cfg.resolution)
        y_coords = np.arange(0.0, self.world_cfg.height + self.grid_cfg.resolution, self.grid_cfg.resolution)

        xx, yy = np.meshgrid(x_coords, y_coords)
        values = self.noise.sample(
            xx,
            yy,
            scale=self.noise_cfg.scale,
            octaves=self.noise_cfg.octaves,
            persistence=self.noise_cfg.persistence,
            lacunarity=self.noise_cfg.lacunarity,
        )
        density = self._normalize(values)
        return DensityFieldResult(x_coords=x_coords, y_coords=y_coords, density=density)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        v_min = float(values.min())
        v_max = float(values.max())

        if math.isclose(v_min, v_max):
            return np.full_like(values, self.noise_cfg.normalize_min)

        normalized = (values - v_min) / (v_max - v_min)
        return self.noise_cfg.normalize_min + normalized * (
            self.noise_cfg.normalize_max - self.noise_cfg.normalize_min
        )
