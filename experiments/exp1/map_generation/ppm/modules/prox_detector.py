"""Compute simulated 2D proximity returns from SDF obstacle geometry."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from modules.load_world import Obstacle, is_wall_obstacle_name

@dataclass(frozen=True)
class ProximityHit:
    name: str
    qx: float
    qy: float
    dist: float
    bearing_rad: float
    delta_rad: float

@dataclass(frozen=True)
class SensorProx:
    count: int
    sensor_xyyaw: Tuple[float, float, float]
    prox: List[ProximityHit]

class ProxDetector:
    def __init__(self, 
                 obstacle_name_prefixes: Tuple[str, ...] = ("box_", "cylinder_", "wall_", "outer_wall_", "obstacle_"),
                 obstacles: List[Obstacle] = [],
                 eps: float = 1e-6):
        self._prefixes = obstacle_name_prefixes
        self._obstacles = obstacles
        self._eps = eps
        # Pre-merge wall blocks to avoid repeating the same work per sensor.
        self._merged_obstacles = self._merge_wall_blocks_to_macro_walls(
            self._obstacles,
        )

    # geometry helpers
    def _wrap_to_pi(self, a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _bearing(self, px: float, py: float, qx: float, qy: float) -> float:
        return math.atan2(qy - py, qx - px)
    
    def _rot_world_to_local(
            self, x: float, y: float, cx: float, cy: float, yaw: float
    ) -> Tuple[float, float]:
        dx = x - cx
        dy = y - cy
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        lx = cos_yaw * dx - sin_yaw * dy
        ly = sin_yaw * dx + cos_yaw * dy
        return lx, ly
    
    def _rot_local_to_world(
            self, px: float, py: float, cx: float, cy: float, yaw: float
    ) -> Tuple[float, float]:
        c = math.cos(yaw)
        s = math.sin(yaw)
        wx = c * px - s * py + cx
        wy = s * px + c * py + cy
        return wx, wy
    
    def _closest_point_on_circle_surface(
        self, px: float, py: float, cx: float, cy: float, r: float
    ) -> Tuple[float, float]:
        vx = px - cx
        vy = py - cy
        n = math.hypot(vx, vy)
        if n < self._eps:
            # arbitrary direction if sensor at center
            return cx + r, cy
        ux, uy = vx / n, vy / n
        return cx + r * ux, cy + r * uy

    def _ray_hit_circle(
            self,
            px: float, py: float,
            dx: float, dy: float,
            cx: float, cy: float, r: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Ray-circle intersection in 2D. Returns (t, qx, qy) for the smallest t>=0 hit.
        """
        # solve quadratic equation
        ox = px - cx
        oy = py - cy
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        t = None
        if t1 >= 0.0 and t2 >= 0.0:
            t = min(t1, t2)
        elif t1 >= 0.0:
            t = t1
        elif t2 >= 0.0:
            t = t2
        else:
            return None

        qx = px + t * dx
        qy = py + t * dy
        return t, qx, qy
    
    def _ray_hit_aabb_local(
            self,
            px: float, py: float,
            dx: float, dy: float,
            hx: float, hy: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Ray-AABB intersection in local 2D. Returns (t_enter, x_hit, y_hit) for the first intersection.
        """
        tmin = -math.inf
        tmax = math.inf

        if abs(dx) < self._eps:
            if px < -hx or px > hx:
                return None
        else:
            tx1 = (-hx - px) / dx
            tx2 = ( hx - px) / dx
            t1 = min(tx1, tx2)
            t2 = max(tx1, tx2)
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
        
        if abs(dy) < self._eps:
            if py < -hy or py > hy:
                return None
        else:
            ty1 = (-hy - py) / dy
            ty2 = ( hy - py) / dy
            t1 = min(ty1, ty2)
            t2 = max(ty1, ty2)
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)

        if tmax < tmin:
            return None
        
        t_enter = tmin if tmin >= 0.0 else tmax
        if t_enter < 0.0:
            return None
        
        xh = px + t_enter * dx
        yh = py + t_enter * dy
        return t_enter, xh, yh
    
    def _ray_hit_obb(
            self,
            px: float, py: float,
            dx: float, dy: float,
            o: Obstacle,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Ray-OBB intersection by transforming the ray into box local frame and using AABB hit.
        """
        lpx, lpy = self._rot_world_to_local(px, py, o.cx, o.cy, o.yaw)

        c = math.cos(-o.yaw)
        s = math.sin(-o.yaw)
        ldx = c * dx - s * dy
        ldy = s * dx + c * dy

        hit = self._ray_hit_aabb_local(lpx, lpy, ldx, ldy, o.hx, o.hy)
        if hit is None:
            return None
        
        t, lx, ly = hit
        qx, qy = self._rot_local_to_world(lx, ly, o.cx, o.cy, o.yaw)
        return t, qx, qy
    
    def _obb_corners_world(self, o: Obstacle) -> List[Tuple[float, float]]:
        local = [(-o.hx, -o.hy), ( o.hx, -o.hy),
                 ( o.hx,  o.hy), (-o.hx,  o.hy)]
        corners = []
        for lx, ly in local:
            wx, wy = self._rot_local_to_world(lx, ly, o.cx, o.cy, o.yaw)
            corners.append((wx, wy))
        return corners
    
    def _merge_wall_blocks_to_macro_walls(
            self,
            obstacles: List[Obstacle],
    ) -> List[Obstacle]:
        """
        replace many small wall blocks with 4 macro walls.
        """
        wall_blocks = [o for o in obstacles if is_wall_obstacle_name(o.name) and o.kind == "box"]
        non_walls = [o for o in obstacles if not (is_wall_obstacle_name(o.name) and o.kind == "box")]

        if len(wall_blocks) == 0:
            print("Warning: No wall blocks found for merging.")
            return obstacles
        
        xs: List[float] = []
        ys: List[float] = []
        thickness_samples: List[float] = []

        for o in wall_blocks:
            for x, y in self._obb_corners_world(o):
                xs.append(x)
                ys.append(y)
            thickness_samples.append(2.0 * min(o.hx, o.hy))
        
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        t = statistics.median(thickness_samples)
        t = max(float(t), self._eps)
        half_t = 0.5 * t

        cx_mid = 0.5 * (minx + maxx)
        cy_mid = 0.5 * (miny + maxy)
        hx_all = 0.5 * (maxx - minx)
        hy_all = 0.5 * (maxy - miny)

        left = Obstacle(name="macro_wall_left", kind="box",
                        cx=minx + half_t, cy=cy_mid, cz=0.0, yaw=0.0,
                        hx=half_t, hy=hy_all)
        right = Obstacle(name="macro_wall_right", kind="box",
                         cx=maxx - half_t, cy=cy_mid, cz=0.0, yaw=0.0,
                         hx=half_t, hy=hy_all)
        bottom = Obstacle(name="macro_wall_bottom", kind="box",
                          cx=cx_mid, cy=miny + half_t, cz=0.0, yaw=0.0,
                          hx=hx_all, hy=half_t)
        top = Obstacle(name="macro_wall_top", kind="box",
                       cx=cx_mid, cy=maxy - half_t, cz=0.0, yaw=0.0,
                       hx=hx_all, hy=half_t)

        # Keep original wall blocks so interior maze walls remain hittable, and add macro
        # walls as cheap outer-shell proxies to keep performance reasonable.
        return non_walls + wall_blocks + [left, right, bottom, top]
    
    def _raycast_first_hit(
            self,
            px: float, py: float,
            dx: float, dy: float,
            obstacles: List[Obstacle],
            max_range_m: float,
    ) -> Optional[Tuple[Obstacle, float, float, float]]:
        """
        Return the nearest intersection among all obstacles.
        """
        best_o: Optional[Obstacle] = None
        best_t: float = math.inf
        best_qx: float = 0.0
        best_qy: float = 0.0

        for o in obstacles:
            if o.kind == "box":
                hit = self._ray_hit_obb(px, py, dx, dy, o)
            elif o.kind == "cylinder":
                hit = self._ray_hit_circle(px, py, dx, dy, o.cx, o.cy, o.r)

            if hit is None:
                continue

            t, qx, qy = hit
            if t < self._eps:
                continue
            if t > max_range_m:
                continue
            if t < best_t:
                best_t = t
                best_o = o
                best_qx = qx
                best_qy = qy

        if best_o is None:
            return None
        return best_o, best_t, best_qx, best_qy
    
    def _visible_surface_prox_per_obstacle(
            self,
            sensor_xyyaw: Tuple[float, float, float],
            obstacles: List[Obstacle],
            max_range_m: float,
            fov_deg: float,
            angle_step_deg: float = 1.0,
    ) -> List[ProximityHit]:
        px, py, psi = sensor_xyyaw

        # FOV handling
        if fov_deg >= 360.0:
            half = math.pi
        else:
            half = 0.5 * math.radians(fov_deg)

        step = math.radians(max(float(angle_step_deg), 1e-6))

        best: Dict[str, ProximityHit] = {}

        # centered sweep: psi + delta
        k_min = int(math.floor(-half / step))
        k_max = int(math.ceil( half / step))

        for k in range(k_min, k_max + 1):
            delta = k * step
            if abs(delta) > half + self._eps:
                continue

            theta = psi + delta
            dx = math.cos(theta)
            dy = math.sin(theta)

            first = self._raycast_first_hit(px, py, dx, dy, obstacles, max_range_m)
            if first is None:
                continue

            o, dist, qx, qy = first
            bearing = self._bearing(px, py, qx, qy)
            delta_wrapped = self._wrap_to_pi(bearing - psi)

            # Keep the closest hit for each obstacle
            prev = best.get(o.name)
            hit = ProximityHit(
                name=o.name,
                qx=qx,
                qy=qy,
                dist=dist,
                bearing_rad=bearing,
                delta_rad=delta_wrapped,
            )
            if prev is None or hit.dist < prev.dist:
                best[o.name] = hit

        hits = list(best.values())
        hits.sort(key=lambda h: h.dist)
        return hits
    
    def detect(
            self,
            sensor_xyyaw: Tuple[float, float, float],
            max_range_m: float,
            fov_deg: float,
            angle_step_deg: float = 0.5,
    ) -> List[ProximityHit]:
        
        hits = self._visible_surface_prox_per_obstacle(
            sensor_xyyaw=sensor_xyyaw,
            obstacles=self._merged_obstacles,
            max_range_m=max_range_m,
            fov_deg=fov_deg,
            angle_step_deg=angle_step_deg,
        )

        return hits

