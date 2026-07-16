from __future__ import annotations

from typing import Tuple

import numpy as np


def proximity_halfspaces(obs: np.ndarray, prox_pts: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Proximity-point tangent halfspaces (free-space side):

        n = p - o
        n · x <= n · p + eps

    Convert to A x <= b:
        A = n
        b = n·p + eps
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(1, -1)
    prox = np.asarray(prox_pts, dtype=np.float64)
    if prox.size == 0:
        dim = int(obs.shape[1])
        return np.zeros((0, dim), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    N = prox - obs
    b = np.einsum("ij,ij->i", N, prox) + float(eps)
    return N.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
