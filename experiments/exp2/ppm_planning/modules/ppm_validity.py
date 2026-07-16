from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from modules.graph_manager import GraphManager


@dataclass(frozen=True)
class PPMBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


class PPMValidityChecker:
    """PPM free-space validity checker for sampling-based planners."""

    def __init__(
        self,
        graph_manager: GraphManager,
        point_tol: float = 1.0e-8,
        edge_tol: float = 0.0,
        max_sample_attempts: int = 10000,
    ) -> None:
        self.graph_manager = graph_manager
        self.point_tol = float(point_tol)
        self.edge_tol = float(edge_tol)
        self.max_sample_attempts = int(max_sample_attempts)

        self.node_cache: dict[int, Any] = {}
        self.node_positions: dict[int, np.ndarray] = {}
        self.halfplanes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.bounds: PPMBounds | None = None

        self._node_ids: list[int] = []
        self._position_matrix: np.ndarray | None = None
        self._kdtree: cKDTree | None = None

    def build(self) -> None:
        nodes = list(self.graph_manager.get_all_nodes())
        if not nodes:
            raise RuntimeError("PPM graph has no nodes.")

        self.node_cache = {int(node_id): node_data for node_id, node_data in nodes}
        self.node_positions = {
            node_id: np.array([node_data["loc_x"], node_data["loc_y"]], dtype=float)
            for node_id, node_data in self.node_cache.items()
        }
        self.halfplanes = {
            node_id: self._halfplane_params(node_data)
            for node_id, node_data in self.node_cache.items()
        }

        self._node_ids = list(self.node_cache.keys())
        self._position_matrix = np.array(
            [self.node_positions[node_id] for node_id in self._node_ids],
            dtype=float,
        )
        self._kdtree = cKDTree(self._position_matrix)
        self.bounds = self._build_sampling_bounds()

    def is_point_free(self, point: tuple[float, float]) -> bool:
        self._require_built()
        node_id = self._nearest_node_id(point)
        if node_id is None:
            return False
        return self._is_point_in_halfplanes(point, node_id)

    def is_segment_free(self, p0: tuple[float, float], p1: tuple[float, float]) -> bool:
        self._require_built()
        node0_id = self._nearest_node_id(p0)
        node1_id = self._nearest_node_id(p1)
        if node0_id is None or node1_id is None:
            return False

        if not self._is_point_in_halfplanes(p0, node0_id):
            return False
        if not self._is_point_in_halfplanes(p1, node1_id):
            return False

        return self._is_edge_like_polygon_map_freearea(p0, p1, node0_id, node1_id)

    def sample_free_point(self, rng: np.random.Generator) -> tuple[float, float]:
        self._require_built()
        if self.bounds is None:
            raise RuntimeError("PPM sampling bounds are not available.")

        low = np.array([self.bounds.min_x, self.bounds.min_y], dtype=float)
        high = np.array([self.bounds.max_x, self.bounds.max_y], dtype=float)
        for _ in range(self.max_sample_attempts):
            point_arr = rng.uniform(low=low, high=high)
            point = (float(point_arr[0]), float(point_arr[1]))
            if self.is_point_free(point):
                return point
        raise RuntimeError("Failed to sample a PPM free-space point within the attempt limit.")

    def _require_built(self) -> None:
        if self._kdtree is None or self._position_matrix is None or self.bounds is None:
            raise RuntimeError("PPMValidityChecker.build() must be called before use.")

    def _build_sampling_bounds(self) -> PPMBounds:
        min_x_values: list[float] = []
        max_x_values: list[float] = []
        min_y_values: list[float] = []
        max_y_values: list[float] = []

        tmp_data = self.graph_manager.tmp_data
        if tmp_data is not None and tmp_data.size > 0:
            loc_x = np.asarray(tmp_data["loc_x"], dtype=float)
            loc_y = np.asarray(tmp_data["loc_y"], dtype=float)
            finite_mask = np.isfinite(loc_x) & np.isfinite(loc_y)
            if np.any(finite_mask):
                min_x_values.append(float(np.min(loc_x[finite_mask])))
                max_x_values.append(float(np.max(loc_x[finite_mask])))
                min_y_values.append(float(np.min(loc_y[finite_mask])))
                max_y_values.append(float(np.max(loc_y[finite_mask])))

        min_x, max_x, min_y, max_y = self.graph_manager.get_landmark_range()
        bounds_arr = np.array([min_x, max_x, min_y, max_y], dtype=float)
        if np.isfinite(bounds_arr).all():
            min_x_values.append(float(min_x))
            max_x_values.append(float(max_x))
            min_y_values.append(float(min_y))
            max_y_values.append(float(max_y))

        if not min_x_values:
            raise RuntimeError("PPM sampling bounds could not be computed.")
        return PPMBounds(
            min_x=min(min_x_values),
            max_x=max(max_x_values),
            min_y=min(min_y_values),
            max_y=max(max_y_values),
        )

    def _nearest_node_id(self, point: tuple[float, float]) -> int | None:
        if self._kdtree is None:
            return None
        point_vec = np.asarray(point, dtype=float)
        if point_vec.shape != (2,) or not np.isfinite(point_vec).all():
            return None
        _, index = self._kdtree.query(point_vec)
        return self._node_ids[int(index)]

    def _halfplane_params(self, node_data: Any) -> tuple[np.ndarray, np.ndarray]:
        loc = np.array([node_data["loc_x"], node_data["loc_y"]], dtype=float)
        landmarks = node_data["landmarks"]
        landmarks_arr = landmarks.as_array()
        if landmarks_arr.size == 0:
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

        prox = np.column_stack((landmarks_arr["prox_x"], landmarks_arr["prox_y"])).astype(float, copy=False)
        normals = prox - loc
        offsets = np.einsum("ij,ij->i", normals, prox)
        flip_mask = normals @ loc > offsets
        normals[flip_mask] *= -1.0
        offsets[flip_mask] *= -1.0
        return normals, offsets

    def _is_point_in_halfplanes(self, point: tuple[float, float], node_id: int) -> bool:
        normals, offsets = self.halfplanes.get(node_id, (None, None))
        if normals is None or offsets is None or normals.size == 0:
            return False
        point_vec = np.asarray(point, dtype=float)
        if point_vec.shape != (2,) or not np.isfinite(point_vec).all():
            return False
        violations = normals.astype(float) @ point_vec - offsets.astype(float)
        return bool(np.all(violations <= self.point_tol))

    def _is_in_freearea_strict(self, point: tuple[float, float], node_id: int) -> bool:
        """Port of old PolygonMapFreearea.is_in_freearea() strict dot-product check."""
        node_data = self.node_cache.get(node_id)
        if node_data is None:
            return False

        obs = np.array([node_data["loc_x"], node_data["loc_y"]], dtype=float)
        landmarks = node_data["landmarks"]
        landmarks_arr = landmarks.as_array()
        if landmarks_arr.size == 0:
            return False

        prox = np.column_stack((landmarks_arr["prox_x"], landmarks_arr["prox_y"])).astype(float, copy=False)
        point_vec = np.asarray(point, dtype=float)
        if point_vec.shape != (2,) or not np.isfinite(point_vec).all():
            return False

        dot_products = np.einsum("ij,ij->i", prox - obs, prox - point_vec)
        return bool(np.all(dot_products > self.edge_tol))

    def _is_edge_like_polygon_map_freearea(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        node0_id: int,
        node1_id: int,
    ) -> bool:
        """Port of old PolygonMapFreearea.is_edge() from ppm_planner_gui/src/graph_manager.py."""
        p0_in_node1 = self._is_in_freearea_strict(p0, node1_id)
        p1_in_node0 = self._is_in_freearea_strict(p1, node0_id)
        return bool(p0_in_node1 and p1_in_node0)

