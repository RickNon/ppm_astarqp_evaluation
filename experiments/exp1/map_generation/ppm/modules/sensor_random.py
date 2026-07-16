"""Sample PPM sensor poses in free space using random or Halton candidates."""

from __future__ import annotations

import random
import math

from modules.load_world import Obstacle
from modules.prox_detector import ProxDetector, SensorProx
from modules.build_ppm import (
    PPMBuilder2D,
    build_prox_pointset,
    build_sensor_poses,
)
import numpy as np

def sample_sensor_positions(
    obstacles: list[Obstacle],
    num_samples: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    seed: int,
    max_range_m: float,
    fov_deg: float,
    sampling_method: str = "random",
    halton_scramble: bool = True,
    angle_step_deg: float = 2.0,
    eps: float = 1e-3,
) -> list[tuple[float, float, float]]:

    def _point_in_box_local(x_world: float, y_world: float, box: Obstacle, eps: float = 1e-3) -> bool:
        # Transform the world point into the box local frame to handle yawed boxes.
        dx = x_world - box.cx
        dy = y_world - box.cy
        cos_yaw = math.cos(-box.yaw)
        sin_yaw = math.sin(-box.yaw)
        x_local = dx * cos_yaw - dy * sin_yaw
        y_local = dx * sin_yaw + dy * cos_yaw
        return (-box.hx - eps <= x_local <= box.hx + eps) and (-box.hy - eps <= y_local <= box.hy + eps)

    method = sampling_method.lower().strip()
    if method not in {"random", "halton"}:
        raise ValueError(f"Unsupported sampling_method: {sampling_method}")

    # Use a local RNG to avoid mutating the global random state.
    seed_value = seed
    if seed_value == 0:
        # Seed 0 means generate a random seed value.
        seed_value = random.SystemRandom().randrange(1, 2**32)
        print(f"[Info] sensor_random: using random seed {seed_value}")
    rng = random.Random(seed_value)

    halton_sampler = None
    if method == "halton":
        from scipy.stats import qmc

        # The seed only affects Halton when scrambling is enabled.
        halton_sampler = qmc.Halton(
            d=2,
            scramble=halton_scramble,
            seed=seed_value if halton_scramble else None,
        )

    def _next_candidate() -> tuple[float, float]:
        if method == "random":
            return (
                rng.uniform(x_range[0], x_range[1]),
                rng.uniform(y_range[0], y_range[1]),
            )

        assert halton_sampler is not None
        point = halton_sampler.random(n=1)[0]
        x = x_range[0] + point[0] * (x_range[1] - x_range[0])
        y = y_range[0] + point[1] * (y_range[1] - y_range[0])
        return float(x), float(y)

    samples: list[tuple[float, float, float]] = []
    sensor_prox: list[SensorProx] = []
    prox_detector = ProxDetector(obstacles=obstacles)
    ppm_builder = PPMBuilder2D()
    bounds_min = (x_range[0], y_range[0])
    bounds_max = (x_range[1], y_range[1])
    ite = 0
    consecutive_rejections = 0
    while ite < num_samples:
        x, y = _next_candidate()
        z = 0.0
        if any(
            (o.kind == "box" and _point_in_box_local(x, y, o, eps=0.05)) or
            (o.kind == "cylinder" and ((x - o.cx) ** 2 + (y - o.cy) ** 2 <= (o.r + 0.05) ** 2))
            for o in obstacles
        ):
            continue
        if samples:
            # reject if inside any free polygon.
            sensors = build_sensor_poses(samples)
            prox_points = build_prox_pointset(sensor_prox)
            cells = ppm_builder.build_halfspaces(
                sensors=sensors,
                prox_points=prox_points,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                eps=eps,
                include_aabb=True,
            )
            candidate = np.array([x, y], dtype=np.float64)
            if any(
                np.all(cell.halfspaces.A @ candidate <= cell.halfspaces.b + eps)
                for cell in cells.values()
            ):
                consecutive_rejections += 1
                if consecutive_rejections > 1000:
                    print(
                        f"breaking after {ite} iterations with "
                        f"{consecutive_rejections} consecutive rejections"
                    )
                    break
                continue
        consecutive_rejections = 0

        prox_hits = prox_detector.detect(
            sensor_xyyaw=(x, y, z),
            max_range_m=max_range_m,
            fov_deg=fov_deg,
            angle_step_deg=angle_step_deg,
        )
        sensor_prox.append(
            SensorProx(count=len(samples), sensor_xyyaw=(x, y, z), prox=prox_hits)
        )
        ite += 1
        samples.append((x, y, z))

    return samples

