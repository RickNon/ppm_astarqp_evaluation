from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config_loader import MazeWorldConfig
from .maze_geometry import WallBox


class SdfWorldWriter:
    """Write a maze world as a complete SDF document."""

    def __init__(self, config: MazeWorldConfig) -> None:
        self.config = config

    def write(self, walls: Iterable[WallBox], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wall_models = "\n".join(self._serialize_wall(wall) for wall in walls)

        xml = f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="{self.config.world.name}">
    <gravity>0 0 -9.8</gravity>
    <include>
      <uri>model://sun</uri>
    </include>

{self._serialize_ground()}

{wall_models}
  </world>
</sdf>
"""
        path.write_text(xml, encoding="utf-8")
        return path

    def _serialize_ground(self) -> str:
        world = self.config.world
        z = -0.5 * world.ground_thickness
        return f"""    <model name="ground_plane_custom">
      <static>true</static>
      <pose>{0.5 * world.width:.6f} {0.5 * world.height:.6f} {z:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{world.width:.6f} {world.height:.6f} {world.ground_thickness:.6f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{world.width:.6f} {world.height:.6f} {world.ground_thickness:.6f}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>"""

    @staticmethod
    def _serialize_wall(wall: WallBox) -> str:
        z = 0.5 * wall.height
        return f"""    <model name="{wall.name}">
      <static>true</static>
      <pose>{wall.center_x:.6f} {wall.center_y:.6f} {z:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{wall.size_x:.6f} {wall.size_y:.6f} {wall.height:.6f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{wall.size_x:.6f} {wall.size_y:.6f} {wall.height:.6f}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>"""

