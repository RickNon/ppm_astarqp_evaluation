from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError


def delaunay_neighbors(points: np.ndarray, dim: int) -> List[np.ndarray]:
    """Return Delaunay neighbor indices for each site.

    - 2D: neighbors induced by triangle simplices
    - 3D: neighbors induced by tetrahedra simplices

    This avoids constructing a full Voronoi diagram while yielding the exact neighbor graph from Delaunay simplices.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != dim:
        raise ValueError(f"points must be (N,{dim}).")
    n = int(pts.shape[0])
    
    # Handle small number of points (no Delaunay possible).
    if n <= dim + 1:
        out: List[np.ndarray] = []
        for i in range(n):
            out.append(np.asarray([j for j in range(n) if j != i], dtype=np.int32))
        return out
    
    neigh_sets: List[set[int]] = [set() for _ in range(n)]

    try:

        # NOTE: Use QJ to reduce degeneracy failures (coplanar/collinear-ish).
        tri = Delaunay(pts, qhull_options="QJ")
        simplices = tri.simplices

        for s in simplices:
            idx = list(map(int, s))
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = idx[i], idx[j]
                    neigh_sets[a].add(b)
                    neigh_sets[b].add(a)

        return [np.fromiter(s, dtype=np.int32) for s in neigh_sets]
    
    except QhullError:
        out: List[np.ndarray] = []
        for i in range(n):
            out.append(np.asarray([j for j in range(n) if j != i], dtype=np.int32))
        return out


def voronoi_halfspaces_for_site(p: np.ndarray, neighbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Voronoi halfspaces from neighbors q:

        (q - p) · x <= (|q|^2 - |p|^2)/2

    Convert to A x <= b:
        A = (q - p)
        b = (|q|^2 - |p|^2)/2
    """
    p = np.asarray(p, dtype=np.float64).reshape(1, -1)
    if neighbors.size == 0:
        dim = int(p.shape[1])
        return np.zeros((0, dim), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    q = np.asarray(neighbors, dtype=np.float64)
    A = q - p
    b = 0.5 * (np.einsum("ij,ij->i", q, q) - float(np.dot(p.reshape(-1), p.reshape(-1))))
    return A.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
