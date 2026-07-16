from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.spatial import Voronoi
import scipy.sparse as sp
import osqp

from modules.graph_manager import GraphManager


@dataclass
class PlanningResult:
    success: int
    time_ms: float
    path_length: float | None
    path_xy: np.ndarray | None
    error: str | None = None


class VoronoiClass:
    def __init__(self, nodes, bounds):
        self.nodes = nodes
        loc_coords = []
        for node in nodes:
            loc_coords.append([node[0], node[1]["loc_x"], node[1]["loc_y"]])
        self.loc_coords = np.array(loc_coords)
        self.bounds = bounds

    def _clip_segment_to_bbox(self, p0, p1, xmin, xmax, ymin, ymax):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0

        def _update(p, q, t_start, t_end):
            if abs(p) < 1e-12:
                if q < 0:
                    return None, None
                return t_start, t_end
            t = q / p
            if p < 0.0:
                if t > t_end:
                    return None, None
                if t > t_start:
                    t_start = t
            else:
                if t < t_start:
                    return None, None
                if t < t_end:
                    t_end = t
            return t_start, t_end

        t0, t1 = _update(-dx, -(xmin - x0), t0, t1)
        if t0 is None:
            return None
        t0, t1 = _update(dx, xmax - x0, t0, t1)
        if t0 is None:
            return None
        t0, t1 = _update(-dy, -(ymin - y0), t0, t1)
        if t0 is None:
            return None
        t0, t1 = _update(dy, ymax - y0, t0, t1)
        if t0 is None:
            return None
        if t0 > t1:
            return None

        q0 = np.array([x0 + t0 * dx, y0 + t0 * dy], dtype=float)
        q1 = np.array([x0 + t1 * dx, y0 + t1 * dy], dtype=float)
        return q0, q1

    def voronoi_edges(self, ray_length: float = 1.0e5):
        gn_pnts = np.concatenate(
            [
                self.loc_coords[:, 1:],
                np.array([[10000, 1000], [10000, -10000], [-10000, 0]]),
            ]
        )
        vor = Voronoi(gn_pnts)

        bounds_arr = np.array(self.bounds, dtype=float)
        xmin = float(bounds_arr[:, 0].min())
        xmax = float(bounds_arr[:, 0].max())
        ymin = float(bounds_arr[:, 1].min())
        ymax = float(bounds_arr[:, 1].max())

        edges = []
        for (p_a, p_b), (v_a, v_b) in zip(vor.ridge_points, vor.ridge_vertices):
            if p_a >= len(self.loc_coords) or p_b >= len(self.loc_coords):
                continue

            if v_a != -1 and v_b != -1:
                p0 = vor.vertices[v_a]
                p1 = vor.vertices[v_b]
            else:
                v_f = v_b if v_a == -1 else v_a
                finite_pt = vor.vertices[v_f]
                site_vec = vor.points[p_b] - vor.points[p_a]
                dir_vec = np.array([site_vec[1], -site_vec[0]], dtype=float)
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-12:
                    continue
                dir_vec /= norm
                p0 = finite_pt + dir_vec * ray_length
                p1 = finite_pt - dir_vec * ray_length

            clipped = self._clip_segment_to_bbox(p0, p1, xmin, xmax, ymin, ymax)
            if clipped is None:
                continue
            q0, q1 = clipped
            edges.append(
                {
                    "cells": (int(self.loc_coords[p_a, 0]), int(self.loc_coords[p_b, 0])),
                    "segment": [q0.tolist(), q1.tolist()],
                }
            )
        return edges


class ProximityPolygonClass:
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.adjacent_boundary_points = []
        self.nearest_point_on_boundaries = {}
        self.shared_boundary_segments = {}
        self._node_cache = {}

    def set_node_cache(self, node_cache):
        self._node_cache = node_cache or {}

    def halfplane_params(self, node):
        loc = np.array([node["loc_x"], node["loc_y"]], dtype=np.float32)
        landmarks_arr = node["landmarks"].as_array()
        if landmarks_arr.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        prox = np.column_stack((landmarks_arr["prox_x"], landmarks_arr["prox_y"])).astype(np.float32, copy=False)
        normals = prox - loc
        offsets = np.einsum("ij,ij->i", normals, prox)
        mask = np.dot(normals, loc) > offsets
        normals[mask] *= -1
        offsets[mask] *= -1
        return normals, offsets

    def check_adjacency(self, node1, node2, count_pair, edge, eps=1e-8, halfplanes=None):
        p0 = np.asarray(edge["segment"][0])
        p1 = np.asarray(edge["segment"][1])
        v = p1 - p0
        t_global = [0.0, 1.0]

        if halfplanes is not None:
            normals1, offsets1 = halfplanes.get(count_pair[0], (None, None))
            normals2, offsets2 = halfplanes.get(count_pair[1], (None, None))
        else:
            normals1, offsets1 = self.halfplane_params(node1)
            normals2, offsets2 = self.halfplane_params(node2)

        for normals, offsets in ((normals1, offsets1), (normals2, offsets2)):
            if normals is None or offsets is None or normals.size == 0:
                continue
            num = offsets - normals.dot(p0)
            den = normals.dot(v)
            parallel_mask = np.abs(den) < eps
            if np.any(parallel_mask) and np.any(num[parallel_mask] < 0.0):
                return False
            nonparallel_mask = ~parallel_mask
            if not np.any(nonparallel_mask):
                continue

            den_np = den[nonparallel_mask]
            t_vals = num[nonparallel_mask] / den_np
            t_enter = 0.0
            t_leave = 1.0
            pos_mask = den_np > 0.0
            if np.any(pos_mask):
                t_leave = min(t_leave, np.min(t_vals[pos_mask]))
            neg_mask = den_np < 0.0
            if np.any(neg_mask):
                t_enter = max(t_enter, np.max(t_vals[neg_mask]))
            if t_enter - t_leave > eps:
                return False

            t_global[0] = max(t_global[0], t_enter)
            t_global[1] = min(t_global[1], t_leave)
            if t_global[0] - t_global[1] > eps:
                return False

        q0 = p0 + t_global[0] * v
        q1 = p0 + t_global[1] * v
        node1_count, node2_count = count_pair[0], count_pair[1]
        self.adjacent_boundary_points.append(
            [(min(node1_count, node2_count), max(node1_count, node2_count)), q0, q1]
        )
        return True

    def get_projection_param(self, endpoint1, endpoint2, mid_point):
        x1, y1 = endpoint1
        x2, y2 = endpoint2
        xm, ym = mid_point
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            return 0.0
        t = ((xm - x1) * dx + (ym - y1) * dy) / denom
        return max(0.0, min(1.0, t))

    def get_nearest_point_on_boundaries(self):
        self.nearest_point_on_boundaries = {}
        self.shared_boundary_segments = {}
        for adjacent_segment in self.adjacent_boundary_points:
            count_pair = adjacent_segment[0]
            node1 = self._node_cache.get(count_pair[0])
            node2 = self._node_cache.get(count_pair[1])
            if node1 is None or node2 is None:
                continue
            node1_xy = np.array([node1["loc_x"], node1["loc_y"]], dtype=float)
            node2_xy = np.array([node2["loc_x"], node2["loc_y"]], dtype=float)
            mid_point = (node1_xy + node2_xy) / 2.0
            endpoint1 = adjacent_segment[1]
            endpoint2 = adjacent_segment[2]
            t = self.get_projection_param(endpoint1, endpoint2, mid_point)
            nearest_point = endpoint1 + t * (endpoint2 - endpoint1)
            key = (min(count_pair[0], count_pair[1]), max(count_pair[0], count_pair[1]))
            self.nearest_point_on_boundaries[key] = nearest_point
            self.shared_boundary_segments[key] = (endpoint1, endpoint2)

    def make_poly_graph(self, vor_edges):
        self.adjacent_boundary_points = []
        if not self._node_cache:
            self._node_cache = {node_id: data for node_id, data in self.graph_manager.get_all_nodes()}
        self.poly_graph = {node_id: [] for node_id in self._node_cache}
        halfplanes = {node_id: self.halfplane_params(node_data) for node_id, node_data in self._node_cache.items()}

        for edge in vor_edges:
            count_pair = edge["cells"]
            node1 = self._node_cache.get(count_pair[0])
            node2 = self._node_cache.get(count_pair[1])
            adjacency = self.check_adjacency(node1, node2, count_pair, edge, halfplanes=halfplanes)
            if adjacency:
                self.poly_graph[count_pair[0]].append(count_pair[1])
                self.poly_graph[count_pair[1]].append(count_pair[0])

        self.get_nearest_point_on_boundaries()
        return self.poly_graph


class AstarQPBase:
    """Base A*QP: distance-based A* followed by the boundary QP."""

    def __init__(
        self,
        graph_manager: GraphManager,
        qp_solver: str = "OSQP",
    ) -> None:
        self.graph_manager = graph_manager
        self.qp_solver = qp_solver
        self.prox_poly = ProximityPolygonClass(graph_manager)
        self._node_cache = {}
        self._node_positions = {}

    def make_astar_graph(self) -> None:
        nodes = list(self.graph_manager.get_all_nodes())
        prox_range = self.graph_manager.get_landmark_range_coords()
        self._node_cache = {node_id: data for node_id, data in nodes}
        self._node_positions = {
            node_id: np.array([data["loc_x"], data["loc_y"]], dtype=float)
            for node_id, data in self._node_cache.items()
        }
        self.prox_poly.set_node_cache(self._node_cache)
        voronoi = VoronoiClass(nodes, prox_range)
        vor_edges = voronoi.voronoi_edges()
        self.poly_graph = self.prox_poly.make_poly_graph(vor_edges)

    def heuristic(self, current_id: int, goal_id: int) -> float:
        current_xy = self._node_positions[current_id]
        goal_xy = self._node_positions[goal_id]
        return float(np.linalg.norm(current_xy - goal_xy))

    def _is_point_in_polygon(self, point: tuple[float, float], node_id: int, tol: float = 1.0e-8) -> bool:
        """Check whether a point satisfies the half-space constraints of a polygon node."""
        node = self._node_cache.get(node_id)
        if node is None:
            return False
        normals, offsets = self.prox_poly.halfplane_params(node)
        if normals.size == 0:
            return False
        point_vec = np.asarray(point, dtype=float)
        violations = normals.astype(float) @ point_vec - offsets.astype(float)
        return bool(np.all(violations <= tol))

    def _validate_query_points(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        start_node_id: int,
        goal_node_id: int,
    ) -> None:
        """Reject start or goal points that are outside the nearest free-area polygon."""
        if not self._is_point_in_polygon(start, start_node_id):
            raise RuntimeError("Start point is outside the nearest free-area polygon.")
        if not self._is_point_in_polygon(goal, goal_node_id):
            raise RuntimeError("Goal point is outside the nearest free-area polygon.")

    def _same_polygon_straight_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        start_node_id: int,
        goal_node_id: int,
    ) -> np.ndarray | None:
        """Return a straight path when start and goal are inside the same convex polygon."""
        if start_node_id != goal_node_id:
            return None
        if not self._is_point_in_polygon(start, start_node_id):
            return None
        if not self._is_point_in_polygon(goal, goal_node_id):
            return None
        return np.asarray([start, goal], dtype=float)

    def edge_cost_via_nearest(self, a_id: int, b_id: int) -> float:
        a_xy = self._node_positions[a_id]
        b_xy = self._node_positions[b_id]
        key = (min(a_id, b_id), max(a_id, b_id))
        nearest_map = self.prox_poly.nearest_point_on_boundaries
        if key in nearest_map:
            point = np.array(nearest_map[key], dtype=float)
            return float(np.linalg.norm(a_xy - point) + np.linalg.norm(point - b_xy))
        return float(np.linalg.norm(a_xy - b_xy))

    def astar_search(self, start_id: int, goal_id: int):
        open_set = []
        g_score = {node: float("inf") for node in self.poly_graph}
        f_score = {node: float("inf") for node in self.poly_graph}
        came_from = {}

        g_score[start_id] = 0.0
        f_score[start_id] = self.heuristic(start_id, goal_id)
        heapq.heappush(open_set, (f_score[start_id], start_id))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_id:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self.poly_graph[current]:
                tentative_g_score = g_score[current] + self.edge_cost_via_nearest(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_id)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def qp(self, astar_path, start, goal):
        astar_shared_edges = []
        segment_lookup = self.prox_poly.shared_boundary_segments
        for i in range(len(astar_path) - 1):
            id_1 = astar_path[i]
            id_2 = astar_path[i + 1]
            key = (min(id_1, id_2), max(id_1, id_2))
            if key not in segment_lookup:
                raise RuntimeError(f"Shared boundary segment not found for edge {key}.")
            endpoint1, endpoint2 = segment_lookup[key]
            astar_shared_edges.append([key, np.array(endpoint1), np.array(endpoint2)])

        num_of_waypoints = len(astar_path) + 1
        num_interior = num_of_waypoints - 2
        x_var = cp.Variable((num_of_waypoints, 2))
        t_var = cp.Variable(num_interior)

        objective = cp.Minimize(cp.sum_squares(x_var[1:, :] - x_var[:-1, :]))
        constraints = [x_var[0, :] == np.array(start), x_var[-1, :] == np.array(goal)]
        for i, edge in enumerate(astar_shared_edges):
            endpoint1 = edge[1]
            endpoint2 = edge[2]
            direction = endpoint2 - endpoint1
            if np.linalg.norm(direction) < 1e-6:
                constraints.append(x_var[i + 1, :] == endpoint1)
                continue
            constraints += [
                x_var[i + 1, :] == endpoint1 + t_var[i] * direction,
                t_var[i] >= 0,
                t_var[i] <= 1,
            ]

        if self.qp_solver == "OSQP_NATIVE":
            return self._solve_qp_with_osqp_native(
                astar_shared_edges=astar_shared_edges,
                start=start,
                goal=goal,
            )

        problem = cp.Problem(objective, constraints)
        solver_constant = getattr(cp, self.qp_solver, None)
        if solver_constant is None:
            raise RuntimeError(f"Unsupported QP solver: {self.qp_solver}")
        problem.solve(solver=solver_constant, verbose=False, warm_start=True)

        if x_var.value is None:
            raise RuntimeError("QP failed to produce a valid path.")
        qp_path = np.array(x_var.value, dtype=float)
        if len(qp_path) < 2:
            raise RuntimeError("QP path is not valid.")
        return qp_path

    def _solve_qp_with_osqp_native(self, astar_shared_edges, start, goal):
        num_of_waypoints = len(astar_shared_edges) + 2
        num_interior = num_of_waypoints - 2
        dim_x = 2 * num_of_waypoints
        dim_t = num_interior
        dim_z = dim_x + dim_t

        p_data = []
        p_row = []
        p_col = []
        for k in range(num_of_waypoints - 1):
            for d in range(2):
                i0 = 2 * k + d
                i1 = 2 * (k + 1) + d
                p_row.extend([i0, i0, i1, i1])
                p_col.extend([i0, i1, i0, i1])
                p_data.extend([2.0, -2.0, -2.0, 2.0])
        p_mat = sp.csc_matrix((p_data, (p_row, p_col)), shape=(dim_z, dim_z))
        q_vec = np.zeros(dim_z, dtype=np.float64)

        a_data = []
        a_row = []
        a_col = []
        l_list = []
        u_list = []
        row = 0

        a_row.append(row)
        a_col.append(0)
        a_data.append(1.0)
        l_list.append(float(start[0]))
        u_list.append(float(start[0]))
        row += 1

        a_row.append(row)
        a_col.append(1)
        a_data.append(1.0)
        l_list.append(float(start[1]))
        u_list.append(float(start[1]))
        row += 1

        idx_xn = 2 * (num_of_waypoints - 1)
        idx_yn = 2 * (num_of_waypoints - 1) + 1
        a_row.append(row)
        a_col.append(idx_xn)
        a_data.append(1.0)
        l_list.append(float(goal[0]))
        u_list.append(float(goal[0]))
        row += 1

        a_row.append(row)
        a_col.append(idx_yn)
        a_data.append(1.0)
        l_list.append(float(goal[1]))
        u_list.append(float(goal[1]))
        row += 1

        for i, edge in enumerate(astar_shared_edges):
            endpoint1 = edge[1]
            endpoint2 = edge[2]
            direction = endpoint2 - endpoint1
            idx_wp_x = 2 * (i + 1)
            idx_wp_y = 2 * (i + 1) + 1

            if np.linalg.norm(direction) < 1e-6:
                a_row.append(row)
                a_col.append(idx_wp_x)
                a_data.append(1.0)
                l_list.append(float(endpoint1[0]))
                u_list.append(float(endpoint1[0]))
                row += 1

                a_row.append(row)
                a_col.append(idx_wp_y)
                a_data.append(1.0)
                l_list.append(float(endpoint1[1]))
                u_list.append(float(endpoint1[1]))
                row += 1
                continue

            idx_t = dim_x + i

            a_row.extend([row, row])
            a_col.extend([idx_wp_x, idx_t])
            a_data.extend([1.0, -float(direction[0])])
            l_list.append(float(endpoint1[0]))
            u_list.append(float(endpoint1[0]))
            row += 1

            a_row.extend([row, row])
            a_col.extend([idx_wp_y, idx_t])
            a_data.extend([1.0, -float(direction[1])])
            l_list.append(float(endpoint1[1]))
            u_list.append(float(endpoint1[1]))
            row += 1

            a_row.append(row)
            a_col.append(idx_t)
            a_data.append(1.0)
            l_list.append(0.0)
            u_list.append(np.inf)
            row += 1

            a_row.append(row)
            a_col.append(idx_t)
            a_data.append(1.0)
            l_list.append(-np.inf)
            u_list.append(1.0)
            row += 1

        a_mat = sp.csc_matrix((a_data, (a_row, a_col)), shape=(row, dim_z))
        l_vec = np.array(l_list, dtype=np.float64)
        u_vec = np.array(u_list, dtype=np.float64)

        solver = osqp.OSQP()
        solver.setup(
            P=p_mat,
            q=q_vec,
            A=a_mat,
            l=l_vec,
            u=u_vec,
            verbose=False,
            polish=True,
            eps_abs=1.0e-7,
            eps_rel=1.0e-7,
            max_iter=200000,
        )
        solution = solver.solve()
        if solution.info.status_val not in (1,):
            raise RuntimeError(f"OSQP_NATIVE solver failed with status: {solution.info.status}")

        z_val = solution.x
        qp_path = np.array(z_val[:dim_x].reshape((num_of_waypoints, 2)), dtype=float)
        if len(qp_path) < 2:
            raise RuntimeError("OSQP_NATIVE path is not valid.")
        return qp_path

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanningResult:
        started_at = time.perf_counter()
        try:
            self.make_astar_graph()
            start_node = self.graph_manager.get_nearest_initial_node(start)
            goal_node = self.graph_manager.get_nearest_initial_node(goal)
            if start_node is None or goal_node is None:
                raise RuntimeError("Nearest graph node was not found for start or goal.")

            self._validate_query_points(start, goal, start_node[0], goal_node[0])
            straight_path = self._same_polygon_straight_path(start, goal, start_node[0], goal_node[0])
            if straight_path is not None:
                elapsed = time.perf_counter() - started_at
                return PlanningResult(
                    success=1,
                    time_ms=1000.0 * elapsed,
                    path_length=float(np.linalg.norm(straight_path[1] - straight_path[0])),
                    path_xy=straight_path,
                )

            astar_path = self.astar_search(start_node[0], goal_node[0])
            if astar_path is None:
                raise RuntimeError("A* search failed to find a path.")

            final_path = self.qp(astar_path, start, goal)
            path_length = float(
                sum(np.linalg.norm(final_path[i + 1] - final_path[i]) for i in range(len(final_path) - 1))
            )
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=1,
                time_ms=1000.0 * elapsed,
                path_length=path_length,
                path_xy=final_path,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=0,
                time_ms=1000.0 * elapsed,
                path_length=None,
                path_xy=None,
                error=str(exc),
            )


class _AstarQPShared(AstarQPBase):
    """Shared wide-space search and smoothing implementation."""

    def __init__(self, graph_manager: GraphManager, qp_solver: str = "OSQP", mu: float = 1.0) -> None:
        super().__init__(graph_manager=graph_manager, qp_solver=qp_solver)
        self.mu = float(mu)
        self._node_clearance_median = None
        self._shared_boundary_length_median = None

    def _node_clearance_mean(self, node_id: int) -> float:
        """Calculate mean distance from a node to its proximity points."""
        node = self._node_cache.get(node_id)
        if node is None:
            return 0.0
        loc = np.array([node["loc_x"], node["loc_y"]], dtype=float)
        landmarks = node.get("landmarks")
        if landmarks is None:
            return 0.0
        landmarks_arr = landmarks.as_array()
        if landmarks_arr.size == 0:
            return 0.0
        prox = np.column_stack((landmarks_arr["prox_x"], landmarks_arr["prox_y"]))
        distances = np.linalg.norm(prox - loc[None, :], axis=1)
        return float(np.mean(distances)) if distances.size else 0.0

    def _shared_boundary_length(self, a_id: int, b_id: int) -> float:
        """Return the length of the cached shared-boundary segment."""
        key = (min(a_id, b_id), max(a_id, b_id))
        seg = self.prox_poly.shared_boundary_segments.get(key)
        if seg is None:
            return 0.0
        p0, p1 = seg
        return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))

    def _precompute_wide_space_normalizers(self) -> None:
        """Precompute median normalization factors for the wide-space cost."""
        c_vals = []
        for node_id in self.poly_graph.keys():
            c_val = self._node_clearance_mean(node_id)
            if np.isfinite(c_val):
                c_vals.append(c_val)
        c_vals = np.array(c_vals, dtype=float)
        c_med = float(np.median(c_vals)) if c_vals.size else 1.0
        if c_med <= 1e-9:
            c_med = 1.0

        l_vals = []
        for (_u, _v), (p0, p1) in self.prox_poly.shared_boundary_segments.items():
            length = float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))
            if np.isfinite(length):
                l_vals.append(length)
        l_vals = np.array(l_vals, dtype=float)
        l_med = float(np.median(l_vals)) if l_vals.size else 1.0
        if l_med <= 1e-9:
            l_med = 1.0

        self._node_clearance_median = c_med
        self._shared_boundary_length_median = l_med

    def _wide_space_cost_multiplier(
        self,
        to_node_id: int,
        a_id: int,
        b_id: int,
        eps: float = 1e-2,
    ) -> float:
        """Compute the normalized wide-space cost multiplier."""
        if self._node_clearance_median is None or self._shared_boundary_length_median is None:
            self._precompute_wide_space_normalizers()

        c_hat = self._node_clearance_mean(to_node_id) / float(self._node_clearance_median)
        l_hat = self._shared_boundary_length(a_id, b_id) / float(
            self._shared_boundary_length_median
        )
        if not np.isfinite(c_hat) or c_hat < 0.0:
            c_hat = 0.0
        if not np.isfinite(l_hat) or l_hat < 0.0:
            l_hat = 0.0

        denom = float(eps + c_hat * l_hat)
        if denom <= 1e-9:
            denom = 1e-9
        return 1.0 / denom

    def wide_space_astar_search(self, start_id: int, goal_id: int):
        """Run the wide-space-aware A* search on the polygon graph."""
        open_set = []
        g_score = {node: float("inf") for node in self.poly_graph}
        f_score = {node: float("inf") for node in self.poly_graph}
        came_from = {}

        g_score[start_id] = 0.0
        f_score[start_id] = self.heuristic(start_id, goal_id)
        heapq.heappush(open_set, (f_score[start_id], start_id))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_id:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self.poly_graph[current]:
                neighbor_distance = self.edge_cost_via_nearest(current, neighbor)
                phi = self._wide_space_cost_multiplier(
                    to_node_id=neighbor,
                    a_id=current,
                    b_id=neighbor,
                )
                tentative_g_score = g_score[current] + neighbor_distance * phi
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_id)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _solve_sqp_osqp_native(self, astar_path, astar_shared_bounds, start, goal):
        def idx_x(point_idx: int, dim_idx: int) -> int:
            return 2 * point_idx + dim_idx

        def idx_y(n_x: int, point_idx: int, dim_idx: int) -> int:
            return n_x + 2 * point_idx + dim_idx

        def idx_tau(n_x: int, n_y: int, boundary_idx: int) -> int:
            return n_x + n_y + boundary_idx

        def point_indices(n_x: int, kind: str, idx: int):
            if kind == "X":
                return idx_x(idx, 0), idx_x(idx, 1)
            if kind == "Y":
                return idx_y(n_x, idx, 0), idx_y(n_x, idx, 1)
            raise ValueError(kind)

        def add_quad_term(p_row: list, p_col: list, p_data: list, idxs, coeffs, weight=1.0):
            """Add one quadratic term to the native OSQP objective matrix."""
            w = float(weight)
            for a in range(len(idxs)):
                for b in range(len(idxs)):
                    p_row.append(idxs[a])
                    p_col.append(idxs[b])
                    p_data.append(2.0 * w * float(coeffs[a]) * float(coeffs[b]))
            return p_row, p_col, p_data

        m = len(astar_path)
        b_count = m - 1
        i_count = max(0, m - 2)
        n_x = (m + 1) * 2
        n_y = i_count * 2
        dim = n_x + n_y + b_count

        q_idx = [("X", 0)]
        for j in range(1, m):
            q_idx.append(("X", j))
            if 1 <= j <= m - 2:
                q_idx.append(("Y", j - 1))
        q_idx.append(("X", m))

        p_row = []
        p_col = []
        p_data = []
        for t in range(len(q_idx) - 1):
            kind0, k0 = q_idx[t]
            kind1, k1 = q_idx[t + 1]
            for dim_idx in range(2):
                i0 = point_indices(n_x, kind0, k0)[dim_idx]
                i1 = point_indices(n_x, kind1, k1)[dim_idx]
                p_row, p_col, p_data = add_quad_term(
                    p_row, p_col, p_data, [i1, i0], [1.0, -1.0], weight=1.0
                )
        if self.mu > 0.0:
            for t in range(1, len(q_idx) - 1):
                kind_prev, k_prev = q_idx[t - 1]
                kind_curr, k_curr = q_idx[t]
                kind_next, k_next = q_idx[t + 1]
                for dim_idx in range(2):
                    i_prev = point_indices(n_x, kind_prev, k_prev)[dim_idx]
                    i_curr = point_indices(n_x, kind_curr, k_curr)[dim_idx]
                    i_next = point_indices(n_x, kind_next, k_next)[dim_idx]
                    p_row, p_col, p_data = add_quad_term(
                        p_row,
                        p_col,
                        p_data,
                        [i_next, i_curr, i_prev],
                        [1.0, -2.0, 1.0],
                        weight=self.mu,
                    )
        p_mat = sp.csc_matrix((p_data, (p_row, p_col)), shape=(dim, dim))
        q_vec = np.zeros(dim, dtype=float)

        a_row = []
        a_col = []
        a_data = []
        l_list = []
        u_list = []
        row = 0

        def add_eq(var_idx: int, value: float):
            nonlocal row
            a_row.append(row)
            a_col.append(var_idx)
            a_data.append(1.0)
            l_list.append(float(value))
            u_list.append(float(value))
            row += 1

        add_eq(idx_x(0, 0), start[0])
        add_eq(idx_x(0, 1), start[1])
        add_eq(idx_x(m, 0), goal[0])
        add_eq(idx_x(m, 1), goal[1])

        for j, edge in enumerate(astar_shared_bounds, start=1):
            p0 = np.asarray(edge[1], dtype=float)
            p1 = np.asarray(edge[2], dtype=float)
            d_vec = p1 - p0
            tau_idx = idx_tau(n_x, n_y, j - 1)

            a_row.extend([row, row])
            a_col.extend([idx_x(j, 0), tau_idx])
            a_data.extend([1.0, -float(d_vec[0])])
            l_list.append(float(p0[0]))
            u_list.append(float(p0[0]))
            row += 1

            a_row.extend([row, row])
            a_col.extend([idx_x(j, 1), tau_idx])
            a_data.extend([1.0, -float(d_vec[1])])
            l_list.append(float(p0[1]))
            u_list.append(float(p0[1]))
            row += 1

        if i_count > 0:
            for k in range(i_count):
                poly_id = astar_path[k + 1]
                node = self._node_cache.get(poly_id)
                if node is None:
                    continue
                normals, offsets = self.prox_poly.halfplane_params(node)
                if normals.size == 0:
                    continue
                for nvec, offset in zip(normals, offsets):
                    a_row.extend([row, row])
                    a_col.extend([idx_y(n_x, k, 0), idx_y(n_x, k, 1)])
                    a_data.extend([float(nvec[0]), float(nvec[1])])
                    l_list.append(-np.inf)
                    u_list.append(float(offset))
                    row += 1

        for j in range(b_count):
            tau_idx = idx_tau(n_x, n_y, j)
            a_row.append(row)
            a_col.append(tau_idx)
            a_data.append(1.0)
            l_list.append(0.0)
            u_list.append(np.inf)
            row += 1

            a_row.append(row)
            a_col.append(tau_idx)
            a_data.append(1.0)
            l_list.append(-np.inf)
            u_list.append(1.0)
            row += 1

        a_mat = sp.csc_matrix((a_data, (a_row, a_col)), shape=(row, dim))
        l_vec = np.array(l_list, dtype=float)
        u_vec = np.array(u_list, dtype=float)

        solver = osqp.OSQP()
        solver.setup(
            P=p_mat,
            q=q_vec,
            A=a_mat,
            l=l_vec,
            u=u_vec,
            verbose=False,
            polish=True,
            eps_abs=1.0e-7,
            eps_rel=1.0e-7,
            max_iter=200000,
        )
        solution = solver.solve()
        if solution.info.status_val not in (1,):
            raise RuntimeError(f"OSQP_NATIVE solver failed with status: {solution.info.status}")

        z_val = solution.x
        x_val = z_val[0:n_x].reshape((m + 1, 2))
        y_val = z_val[n_x:n_x + n_y].reshape((i_count, 2)) if i_count > 0 else None

        qp_points = []
        for kind, idx in q_idx:
            if kind == "X":
                qp_points.append(x_val[idx])
            elif kind == "Y":
                if y_val is None:
                    raise RuntimeError("Expected interior waypoints for A*QP (full).")
                qp_points.append(y_val[idx])
            else:
                raise ValueError(kind)
        qp_path = np.asarray(qp_points, dtype=float)
        if len(qp_path) < 2:
            raise RuntimeError("A*QP (full) native path is not valid.")
        return qp_path

    def smoothing_qp(self, astar_path, start, goal):
        """Solve the full A*QP smoothing QP with interior polygon waypoints."""
        astar_shared_bounds = []
        segment_lookup = self.prox_poly.shared_boundary_segments
        for i in range(len(astar_path) - 1):
            id_1 = astar_path[i]
            id_2 = astar_path[i + 1]
            key = (min(id_1, id_2), max(id_1, id_2))
            if key not in segment_lookup:
                raise RuntimeError(f"Shared boundary segment not found for edge {key}.")
            endpoint1, endpoint2 = segment_lookup[key]
            astar_shared_bounds.append([key, np.array(endpoint1, dtype=float), np.array(endpoint2, dtype=float)])

        m = len(astar_path)
        b_count = m - 1
        i_count = max(0, m - 2)
        x_var = cp.Variable((m + 1, 2))
        tau_var = cp.Variable(b_count)
        y_var = cp.Variable((i_count, 2)) if i_count > 0 else None

        q_points = [x_var[0, :]]
        for j in range(1, m):
            q_points.append(x_var[j, :])
            if 1 <= j <= m - 2:
                if y_var is None:
                    raise RuntimeError("Expected interior variable matrix for A*QP (full).")
                q_points.append(y_var[j - 1, :])
        q_points.append(x_var[m, :])

        obj_dist = 0
        for t in range(len(q_points) - 1):
            obj_dist += cp.sum_squares(q_points[t + 1] - q_points[t])

        obj_smooth = 0
        if self.mu > 0.0:
            for t in range(1, len(q_points) - 1):
                obj_smooth += cp.sum_squares(q_points[t + 1] - 2 * q_points[t] + q_points[t - 1])

        objective = cp.Minimize(obj_dist + self.mu * obj_smooth)
        constraints = [
            x_var[0, :] == np.asarray(start, dtype=float),
            x_var[m, :] == np.asarray(goal, dtype=float),
        ]

        for j, edge in enumerate(astar_shared_bounds):
            p0 = edge[1]
            p1 = edge[2]
            d_vec = p1 - p0
            if np.linalg.norm(d_vec) < 1e-9:
                constraints.append(x_var[j + 1, :] == p0)
                constraints.append(tau_var[j] == 0.0)
            else:
                constraints += [
                    x_var[j + 1, :] == p0 + tau_var[j] * d_vec,
                    tau_var[j] >= 0.0,
                    tau_var[j] <= 1.0,
                ]

        if i_count > 0:
            for k in range(i_count):
                poly_id = astar_path[k + 1]
                node = self._node_cache.get(poly_id)
                if node is None:
                    continue
                normals, offsets = self.prox_poly.halfplane_params(node)
                if normals.size == 0:
                    continue
                constraints.append(normals @ y_var[k, :] <= offsets)

        if self.qp_solver == "OSQP_NATIVE":
            return self._solve_sqp_osqp_native(astar_path, astar_shared_bounds, start, goal)

        solver_constant = getattr(cp, self.qp_solver, None)
        if solver_constant is None:
            raise RuntimeError(f"Unsupported QP solver: {self.qp_solver}")
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver_constant, verbose=False, warm_start=True)

        x_val = np.asarray(x_var.value, dtype=float)
        y_val = np.asarray(y_var.value, dtype=float) if i_count > 0 else None
        qp_points = [x_val[0]]
        for j in range(1, m):
            qp_points.append(x_val[j])
            if 1 <= j <= m - 2:
                if y_val is None:
                    raise RuntimeError("Expected interior waypoints for A*QP (full).")
                qp_points.append(y_val[j - 1])
        qp_points.append(x_val[m])
        qp_path = np.asarray(qp_points, dtype=float)
        if len(qp_path) < 2:
            raise RuntimeError("A*QP (full) path is not valid.")
        return qp_path

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanningResult:
        started_at = time.perf_counter()
        try:
            self.make_astar_graph()
            start_node = self.graph_manager.get_nearest_initial_node(start)
            goal_node = self.graph_manager.get_nearest_initial_node(goal)
            if start_node is None or goal_node is None:
                raise RuntimeError("Nearest graph node was not found for start or goal.")

            self._validate_query_points(start, goal, start_node[0], goal_node[0])
            straight_path = self._same_polygon_straight_path(start, goal, start_node[0], goal_node[0])
            if straight_path is not None:
                elapsed = time.perf_counter() - started_at
                return PlanningResult(
                    success=1,
                    time_ms=1000.0 * elapsed,
                    path_length=float(np.linalg.norm(straight_path[1] - straight_path[0])),
                    path_xy=straight_path,
                )

            self._precompute_wide_space_normalizers()
            astar_path = self.wide_space_astar_search(start_node[0], goal_node[0])
            if astar_path is None:
                raise RuntimeError("Wide-space A* search failed to find a path.")

            final_path = self.smoothing_qp(astar_path, start, goal)
            path_length = float(
                sum(np.linalg.norm(final_path[i + 1] - final_path[i]) for i in range(len(final_path) - 1))
            )
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=1,
                time_ms=1000.0 * elapsed,
                path_length=path_length,
                path_xy=final_path,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=0,
                time_ms=1000.0 * elapsed,
                path_length=None,
                path_xy=None,
                error=str(exc),
            )


class AstarQPWideSpace(_AstarQPShared):
    """Wide-space A*QP: wide-space-aware search with the base boundary QP."""

    def __init__(self, graph_manager: GraphManager, qp_solver: str = "OSQP") -> None:
        super().__init__(graph_manager=graph_manager, qp_solver=qp_solver, mu=0.0)

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanningResult:
        """Run A*QP (wide-space): wide-space search with the base boundary QP."""
        started_at = time.perf_counter()
        try:
            self.make_astar_graph()
            start_node = self.graph_manager.get_nearest_initial_node(start)
            goal_node = self.graph_manager.get_nearest_initial_node(goal)
            if start_node is None or goal_node is None:
                raise RuntimeError("Nearest graph node was not found for start or goal.")

            self._validate_query_points(start, goal, start_node[0], goal_node[0])
            straight_path = self._same_polygon_straight_path(start, goal, start_node[0], goal_node[0])
            if straight_path is not None:
                elapsed = time.perf_counter() - started_at
                return PlanningResult(
                    success=1,
                    time_ms=1000.0 * elapsed,
                    path_length=float(np.linalg.norm(straight_path[1] - straight_path[0])),
                    path_xy=straight_path,
                )

            self._precompute_wide_space_normalizers()
            astar_path = self.wide_space_astar_search(start_node[0], goal_node[0])
            if astar_path is None:
                raise RuntimeError("Wide-space A* search failed to find a path.")

            final_path = self.qp(astar_path, start, goal)
            path_length = float(
                sum(np.linalg.norm(final_path[i + 1] - final_path[i]) for i in range(len(final_path) - 1))
            )
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=1,
                time_ms=1000.0 * elapsed,
                path_length=path_length,
                path_xy=final_path,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=0,
                time_ms=1000.0 * elapsed,
                path_length=None,
                path_xy=None,
                error=str(exc),
            )


class AstarQPFull(_AstarQPShared):
    """Full A*QP: wide-space-aware search followed by the smoothing QP."""

