from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def aabb_halfspaces(bmin: Sequence[float], bmax: Sequence[float], dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box [bmin, bmax] as halfspaces A x <= b."""
    bmin = np.asarray(bmin, dtype=np.float64).reshape(dim)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(dim)

    A = []
    b = []
    for k in range(dim):
        e = np.zeros((dim,), dtype=np.float64)
        e[k] = 1.0
        A.append(e.copy())
        b.append(float(bmax[k]))
        A.append((-e).copy())
        b.append(float(-bmin[k]))

    return np.vstack(A), np.asarray(b, dtype=np.float64)
