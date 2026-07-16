from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .core.types import SensorPose, ProximityPointSet
from .core.cells import HalfspacePolytope, PPMCell
from .algorithms.aabb import aabb_halfspaces
from .algorithms.voronoi import delaunay_neighbors, voronoi_halfspaces_for_site
from .algorithms.prox import proximity_halfspaces


@dataclass
class _PPMBuilderBase:
    dim: int

    def build_halfspaces(
        self,
        sensors: Iterable[SensorPose],
        prox_points: ProximityPointSet,
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
        eps: float = 1e-3,
        include_aabb: bool = True,
        meta: dict | None = None,
    ) -> Dict[int, PPMCell]:
        """Build per-sensor PPM cells in halfspace form.

        Output cells represent:
            cell(i) = Voronoi(i) ∩ Prox(i) (∩ AABB)
        """
        eps = float(eps)
        sensor_list = list(sensors)
        if not sensor_list:
            return {}

        ids = np.array([int(s.id) for s in sensor_list], dtype=np.int64)
        obs = np.vstack([s.as_array(self.dim) for s in sensor_list]).astype(np.float64, copy=False)

        A_box, b_box = aabb_halfspaces(bounds_min, bounds_max, dim=self.dim)

        # --- Voronoi constraints (from Delaunay neighbor graph) ---
        vorA: Dict[int, np.ndarray] = {}
        vorb: Dict[int, np.ndarray] = {}
        neigh = delaunay_neighbors(obs, dim=self.dim)

        for i, sid in enumerate(ids):
            nb_idx = neigh[i]
            nb_xyz = obs[nb_idx] if nb_idx.size > 0 else np.zeros((0, self.dim), dtype=np.float64)
            A_i, b_i = voronoi_halfspaces_for_site(obs[i], nb_xyz)
            vorA[int(sid)] = A_i
            vorb[int(sid)] = b_i

        # --- Prox constraints + stack ---
        out: Dict[int, PPMCell] = {}
        for i, sid in enumerate(ids):
            obs_i = obs[i]
            prox_i = prox_points.for_sensor(int(sid), dim=self.dim)
            A_prox, b_prox = proximity_halfspaces(obs_i, prox_i, eps=eps)

            A_voro = vorA.get(int(sid), np.zeros((0, self.dim), dtype=np.float64))
            b_voro = vorb.get(int(sid), np.zeros((0,), dtype=np.float64))

            A_parts: List[np.ndarray] = [A_voro, A_prox]
            b_parts: List[np.ndarray] = [b_voro, b_prox]
            if include_aabb:
                A_parts.append(A_box)
                b_parts.append(b_box)

            A = np.vstack([x for x in A_parts if x.size > 0]).astype(np.float64, copy=False)
            b = np.concatenate([x for x in b_parts if x.size > 0]).astype(np.float64, copy=False)

            poly = HalfspacePolytope(A=A, b=b, dim=self.dim)
            poly.validate()

            out[int(sid)] = PPMCell(
                sensor_id=int(sid),
                sensor_pos=obs_i.copy(),
                halfspaces=poly,
                num_prox=int(prox_i.shape[0]),
                num_voronoi=int(A_voro.shape[0]),
                meta=dict(meta) if meta else None,
            )

        return out
