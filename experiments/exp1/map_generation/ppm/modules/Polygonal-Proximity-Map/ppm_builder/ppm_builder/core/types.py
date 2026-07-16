from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SensorPose:
    """A single node (sensor pose) in the proximity-point pose graph."""

    id: int
    position: np.ndarray  # shape (D,)

    def as_array(self, dim: int) -> np.ndarray:
        # NOTE: Added explicit dimension validation to keep 2D/3D switching safe.
        p = np.asarray(self.position, dtype=np.float64).reshape(-1)
        if p.size != dim:
            raise ValueError(f"SensorPose.position must have dim={dim}, got {p.size}.")
        return p


@dataclass(frozen=True)
class ProximityPointSet:
    """Mapping: sensor_id -> list of proximity points (each is a tuple of floats)."""

    points: Dict[int, List[Tuple[float, ...]]]

    @staticmethod
    def from_dict(d: Mapping[int, Iterable[Sequence[float]]]) -> "ProximityPointSet":
        # NOTE: Added normalization to tuples for stable serialization / hashing.
        out: Dict[int, List[Tuple[float, ...]]] = {}
        for sid, pts in d.items():
            out[int(sid)] = [tuple(float(x) for x in p) for p in pts]
        return ProximityPointSet(points=out)

    def for_sensor(self, sensor_id: int, dim: int) -> np.ndarray:
        pts = self.points.get(int(sensor_id), [])
        if not pts:
            return np.zeros((0, dim), dtype=np.float64)
        arr = np.asarray(pts, dtype=np.float64)
        arr = arr.reshape(-1, dim)  # NOTE: explicit reshape to fail early on wrong dims.
        return arr
