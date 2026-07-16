from __future__ import annotations

from pathlib import Path
from typing import Iterable

from modules.config_loader import GenerationConfig, WorldConfig
from modules.obstacle_generator import ObstacleInstance


class SdfWorldWriter:
    """Write a complete SDF world with outer walls and random obstacles."""

    def __init__(self, world_cfg: WorldConfig, generation_cfg: GenerationConfig) -> None:
        self.world_cfg = world_cfg
        self.generation_cfg = generation_cfg

    def write(self, obstacles: Iterable[ObstacleInstance], output_path: str | Path) -> None:
        path = Path(output_path)
        obstacle_models = "\n".join(self._serialize_obstacle(obstacle) for obstacle in obstacles)
        wall_models = "\n".join(self._serialize_outer_walls())
        ground_model = self._serialize_ground()

        xml = f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="{self.world_cfg.name}">
    <gravity>0 0 -9.8</gravity>
    <include>
      <uri>model://sun</uri>
    </include>

{ground_model}

{wall_models}

{obstacle_models}
  </world>
</sdf>
"""
        path.write_text(xml, encoding="utf-8")

    def _serialize_ground(self) -> str:
        z = -0.5 * self.world_cfg.ground_thickness
        return f"""    <model name="ground_plane_custom">
      <static>true</static>
      <pose>{0.5 * self.world_cfg.width:.6f} {0.5 * self.world_cfg.height:.6f} {z:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{self.world_cfg.width:.6f} {self.world_cfg.height:.6f} {self.world_cfg.ground_thickness:.6f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{self.world_cfg.width:.6f} {self.world_cfg.height:.6f} {self.world_cfg.ground_thickness:.6f}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>"""

    def _serialize_outer_walls(self) -> list[str]:
        t = self.world_cfg.wall_thickness
        h = self.world_cfg.wall_height
        half_h = 0.5 * h
        w = self.world_cfg.width
        d = self.world_cfg.height

        wall_specs = [
            ("outer_wall_bottom", 0.5 * w, -0.5 * t, w + 2.0 * t, t, h),
            ("outer_wall_top", 0.5 * w, d + 0.5 * t, w + 2.0 * t, t, h),
            ("outer_wall_left", -0.5 * t, 0.5 * d, t, d, h),
            ("outer_wall_right", w + 0.5 * t, 0.5 * d, t, d, h),
        ]

        models = []
        for name, x, y, sx, sy, sz in wall_specs:
            models.append(
                f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.6f} {y:.6f} {half_h:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{sx:.6f} {sy:.6f} {sz:.6f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{sx:.6f} {sy:.6f} {sz:.6f}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>"""
            )
        return models

    def _serialize_obstacle(self, obstacle: ObstacleInstance) -> str:
        if obstacle.kind == "box":
            z = 0.5 * float(obstacle.height)
            return f"""    <model name="{obstacle.name}">
      <static>true</static>
      <pose>{obstacle.center_x:.6f} {obstacle.center_y:.6f} {z:.6f} 0 0 {obstacle.yaw:.6f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{obstacle.size_x:.6f} {obstacle.size_y:.6f} {obstacle.height:.6f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{obstacle.size_x:.6f} {obstacle.size_y:.6f} {obstacle.height:.6f}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>"""

        if obstacle.kind == "cylinder":
            z = 0.5 * float(obstacle.height)
            return f"""    <model name="{obstacle.name}">
      <static>true</static>
      <pose>{obstacle.center_x:.6f} {obstacle.center_y:.6f} {z:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>{obstacle.radius:.6f}</radius>
              <length>{obstacle.height:.6f}</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>{obstacle.radius:.6f}</radius>
              <length>{obstacle.height:.6f}</length>
            </cylinder>
          </geometry>
        </visual>
      </link>
    </model>"""

        raise ValueError(f"Unsupported obstacle kind: {obstacle.kind}")
