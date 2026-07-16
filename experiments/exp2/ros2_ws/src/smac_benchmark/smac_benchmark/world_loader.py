from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import cos, sin
from pathlib import Path


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str
    cx: float
    cy: float
    cz: float
    yaw: float
    hx: float = 0.0
    hy: float = 0.0
    r: float = 0.0

    def contains_point(self, px: float, py: float) -> bool:
        if self.kind == "box":
            dx = px - self.cx
            dy = py - self.cy
            c = cos(-self.yaw)
            s = sin(-self.yaw)
            local_x = c * dx - s * dy
            local_y = s * dx + c * dy
            return abs(local_x) <= self.hx and abs(local_y) <= self.hy
        if self.kind == "cylinder":
            return (px - self.cx) ** 2 + (py - self.cy) ** 2 <= self.r ** 2
        raise ValueError(f"Unsupported obstacle kind: {self.kind}")


class WorldLoader:
    def __init__(self, world_file: Path) -> None:
        self.world_file = Path(world_file)

    def _parse_pose(self, pose_str: str) -> tuple[float, float, float, float, float, float]:
        values = [float(value) for value in pose_str.split()]
        if len(values) != 6:
            raise ValueError(f"Invalid pose string: {pose_str}")
        return tuple(values)

    def _get_model_pose_xyyaw(self, model_elem: ET.Element) -> tuple[float, float, float, float]:
        pose_el = model_elem.find("pose")
        if pose_el is None or pose_el.text is None:
            return 0.0, 0.0, 0.0, 0.0
        x, y, z, _roll, _pitch, yaw = self._parse_pose(pose_el.text.strip())
        return x, y, z, yaw

    def _extract_footprint(self, model: ET.Element) -> tuple[str, float, float] | None:
        geometry = model.find("./link/collision/geometry")
        if geometry is None:
            return None

        children = list(geometry)
        if not children:
            return None

        primitive = children[0]
        if primitive.tag == "box":
            size_el = primitive.find("size")
            if size_el is None or size_el.text is None:
                return None
            sx, sy, _sz = [float(value) for value in size_el.text.strip().split()]
            return "box", 0.5 * sx, 0.5 * sy

        if primitive.tag == "cylinder":
            radius_el = primitive.find("radius")
            if radius_el is None or radius_el.text is None:
                return None
            return "cylinder", float(radius_el.text.strip()), 0.0

        return None

    def load_obstacles(
        self,
        obstacle_name_prefixes: tuple[str, ...] = ("box_", "cylinder_", "wall_", "outer_wall_", "obstacle_"),
    ) -> list[Obstacle]:
        tree = ET.parse(self.world_file)
        root = tree.getroot()
        world = root.find("world")
        if world is None:
            raise ValueError("No <world> element found in the SDF file.")

        obstacles: list[Obstacle] = []
        for model in world.findall("model"):
            name = model.get("name", "")
            if not name.startswith(obstacle_name_prefixes):
                continue

            cx, cy, cz, yaw = self._get_model_pose_xyyaw(model)
            footprint = self._extract_footprint(model)
            if footprint is None:
                continue

            if footprint[0] == "box":
                _, hx, hy = footprint
                obstacles.append(Obstacle(name=name, kind="box", cx=cx, cy=cy, cz=cz, yaw=yaw, hx=hx, hy=hy))
            elif footprint[0] == "cylinder":
                _, radius, _ = footprint
                obstacles.append(Obstacle(name=name, kind="cylinder", cx=cx, cy=cy, cz=cz, yaw=yaw, r=radius))

        return obstacles


