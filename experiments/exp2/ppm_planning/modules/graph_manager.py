from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from modules.data_converter import OtoPP, TMP_RECORD_DTYPE


class LandmarkView:
    __slots__ = ("_data", "_slice")

    def __init__(self, data: np.ndarray, row_slice: slice):
        self._data = data
        self._slice = row_slice

    def as_array(self) -> np.ndarray:
        return self._data[self._slice]

    def __len__(self) -> int:
        return self._slice.stop - self._slice.start

    def __bool__(self) -> bool:
        return len(self) > 0


class GraphManager:
    def __init__(self) -> None:
        self.tmp_data: np.ndarray | None = None
        self.graph = nx.Graph()
        self.initial_graph = nx.Graph()
        self.edge_max_distance = 3.0

    def _normalize_tmp_data(self, data: np.ndarray) -> np.ndarray:
        if isinstance(data, np.ndarray) and data.dtype == TMP_RECORD_DTYPE:
            return data
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            return data.astype(TMP_RECORD_DTYPE, copy=False)
        raise TypeError("Unsupported tmp_data format.")

    def load_omnia_files(self, loc_path: str, prox_path: str) -> None:
        converter = OtoPP(loc_path, prox_path)
        self.tmp_data = self._normalize_tmp_data(converter.main_convert())
        self.build_graph(self.tmp_data)
        self.initial_graph = self.graph.copy()

    def build_graph(self, data: np.ndarray) -> None:
        self.graph.clear()
        self.initial_graph.clear()

        tmp_array = self._normalize_tmp_data(data)
        if tmp_array.size == 0:
            return

        counts = tmp_array["count"]
        loc_xs = tmp_array["loc_x"]
        loc_ys = tmp_array["loc_y"]
        prev_count = None
        prev_loc: tuple[float, float] | None = None
        idx = 0

        while idx < len(tmp_array):
            count = int(counts[idx])
            loc = np.array([loc_xs[idx], loc_ys[idx]], dtype=np.float32)
            start_idx = idx
            while idx < len(tmp_array) and int(counts[idx]) == count:
                idx += 1

            landmark_view = LandmarkView(tmp_array, slice(start_idx, idx))
            self.graph.add_node(count, loc_x=loc[0], loc_y=loc[1], landmarks=landmark_view)

            if prev_count is not None and prev_loc is not None:
                dist_prev = math.hypot(loc[0] - prev_loc[0], loc[1] - prev_loc[1])
                if dist_prev < self.edge_max_distance and not self.graph.has_edge(count, prev_count):
                    self.graph.add_edge(count, prev_count, weight=dist_prev)

            prev_count = count
            prev_loc = (loc[0], loc[1])

    def get_all_nodes(self):
        return self.graph.nodes(data=True)

    def get_node(self, count: int):
        return self.graph.nodes[count]

    def get_nearest_initial_node(self, point: tuple[float, float]):
        nearest_node = None
        min_dist = float("inf")
        for node in self.initial_graph.nodes(data=True):
            dist = math.hypot(point[0] - node[1]["loc_x"], point[1] - node[1]["loc_y"])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def get_landmark_range(self) -> tuple[np.float32, np.float32, np.float32, np.float32]:
        if self.tmp_data is None or self.tmp_data.size == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        prox_x = self.tmp_data["prox_x"]
        prox_y = self.tmp_data["prox_y"]
        return (
            np.float32(np.nanmin(prox_x)),
            np.float32(np.nanmax(prox_x)),
            np.float32(np.nanmin(prox_y)),
            np.float32(np.nanmax(prox_y)),
        )

    def get_landmark_range_coords(self) -> list[list[np.float32]]:
        min_x, max_x, min_y, max_y = self.get_landmark_range()
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

    def build_free_area_polygons(self, margin: float = 1.0e-6) -> list[np.ndarray]:
        """Reconstruct bounded convex free-area polygons for all graph nodes."""
        nodes = list(self.get_all_nodes())
        if not nodes:
            return []

        min_x, max_x, min_y, max_y = self.get_landmark_range()
        if not np.isfinite([min_x, max_x, min_y, max_y]).all():
            return []

        node_positions = {
            node_id: np.array([float(data["loc_x"]), float(data["loc_y"])], dtype=float)
            for node_id, data in nodes
        }
        polygons: list[np.ndarray] = []

        for node_id, node_data in nodes:
            loc = node_positions[node_id]
            halfspaces: list[np.ndarray] = []

            # Voronoi halfspaces: keep the side closer to the current node.
            for other_id, other_pos in node_positions.items():
                if other_id == node_id:
                    continue
                normal = other_pos - loc
                offset = 0.5 * (np.dot(other_pos, other_pos) - np.dot(loc, loc))
                halfspaces.append(np.array([normal[0], normal[1], -offset], dtype=float))

            # Proximity halfspaces from the PPM node.
            landmarks_arr = node_data["landmarks"].as_array()
            if landmarks_arr.size > 0:
                prox_points = np.column_stack((landmarks_arr["prox_x"], landmarks_arr["prox_y"])).astype(float, copy=False)
                normals = prox_points - loc
                offsets = np.einsum("ij,ij->i", normals, prox_points)
                flip_mask = normals @ loc > offsets
                normals[flip_mask] *= -1.0
                offsets[flip_mask] *= -1.0
                for normal, offset in zip(normals, offsets):
                    halfspaces.append(np.array([normal[0], normal[1], -float(offset)], dtype=float))

            # Axis-aligned bounding box to keep the intersection bounded.
            halfspaces.extend(
                [
                    np.array([-1.0, 0.0, float(min_x)], dtype=float),   # x >= min_x
                    np.array([1.0, 0.0, -float(max_x)], dtype=float),   # x <= max_x
                    np.array([0.0, -1.0, float(min_y)], dtype=float),   # y >= min_y
                    np.array([0.0, 1.0, -float(max_y)], dtype=float),   # y <= max_y
                ]
            )

            halfspace_array = np.array(halfspaces, dtype=float)
            interior_point = loc.copy()

            values = halfspace_array[:, :2] @ interior_point + halfspace_array[:, 2]
            if np.any(values >= -margin):
                interior_point = interior_point - (np.maximum(values, 0.0).max() + 10.0 * margin)
                # Fallback is only to avoid numerical edge cases; skip the polygon if still infeasible.
                values = halfspace_array[:, :2] @ interior_point + halfspace_array[:, 2]
            if np.any(values >= -margin):
                continue

            try:
                intersection = HalfspaceIntersection(halfspace_array, interior_point)
            except Exception:
                continue

            vertices = np.asarray(intersection.intersections, dtype=float)
            if len(vertices) < 3:
                continue
            try:
                hull = ConvexHull(vertices)
                vertices = vertices[hull.vertices]
            except Exception:
                center = np.mean(vertices, axis=0)
                angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
                vertices = vertices[np.argsort(angles)]

            polygons.append(vertices)

        return polygons

