from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

from ..core.cells import HalfspacePolytope


@dataclass(frozen=True)
class ExtractedPolytope:
    vertices: np.ndarray  # (K, D)
    hull: Optional[ConvexHull]  # 2D/3D hull; None if not computed


def _find_interior_point(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Find a strictly interior point by maximizing slack t (LP).

    Solve:
        maximize t
        s.t. A x + t <= b
             t >= 0
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    dim = int(A.shape[1])
    c = np.zeros((dim + 1,), dtype=np.float64)
    c[-1] = -1.0  # maximize t -> minimize -t

    A_ub = np.hstack([A, np.ones((A.shape[0], 1), dtype=np.float64)])
    bounds = [(None, None)] * dim + [(0.0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
    if (not res.success) or (res.x[-1] <= 1e-8):
        return None
    return np.asarray(res.x[:dim], dtype=np.float64)


def extract_vertices(poly: HalfspacePolytope, qhull_options: str = "QJ") -> Optional[ExtractedPolytope]:
    """Extract vertices for a halfspace polytope (2D or 3D).

    Returns None if:
    - infeasible, unbounded (w.r.t. constraints), or
    - HalfspaceIntersection fails.

    Note
    ----
    This function is intentionally separated from PPM construction, as requested.
    """
    poly.validate()
    A, b = poly.A, poly.b
    interior = _find_interior_point(A, b)
    if interior is None:
        return None

    # SciPy HalfspaceIntersection expects halfspaces in form: a·x + b <= 0
    hs = np.hstack([A, -b.reshape(-1, 1)])

    try:
        hsi = HalfspaceIntersection(hs, interior)
    except Exception:
        return None

    verts = np.asarray(hsi.intersections, dtype=np.float64)
    if verts.shape[0] == 0:
        return None

    hull = None
    try:
        # In 2D: hull.volume is area; in 3D: volume.
        hull = ConvexHull(verts, qhull_options=qhull_options)
    except Exception:
        hull = None

    return ExtractedPolytope(vertices=verts, hull=hull)
