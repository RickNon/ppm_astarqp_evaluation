from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class HalfspacePolytope:
    """Halfspace representation: A x <= b."""

    A: np.ndarray  # (M, D)
    b: np.ndarray  # (M,)
    dim: int

    def validate(self) -> None:
        # validation to prevent silent shape bugs.
        A = np.asarray(self.A, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        if A.ndim != 2:
            raise ValueError("A must be 2D array.")
        if b.ndim != 1:
            raise ValueError("b must be 1D array.")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Inconsistent shapes: A is {A.shape}, b is {b.shape}.")
        if A.shape[1] != self.dim:
            raise ValueError(f"A must have dim={self.dim}, got {A.shape[1]}.")


@dataclass(frozen=True)
class PPMCell:
    """A single PPM free region (per sensor node)."""

    sensor_id: int
    sensor_pos: np.ndarray  # (D,)
    halfspaces: HalfspacePolytope
    num_prox: int
    num_voronoi: int
    meta: Optional[dict] = None
