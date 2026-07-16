from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal


def _is_supported_obstacle_name(name: str, prefixes: Tuple[str, ...]) -> bool:
    # Accept both legacy names and names used by the current map_builder worlds.
    return name.startswith(prefixes)


def is_wall_obstacle_name(name: str) -> bool:
    # Treat both legacy wall blocks and generated outer walls as enclosure walls.
    return name.startswith(("wall_", "outer_wall_", "macro_wall_", "maze_wall_"))

@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: Literal["box", "cylinder"]
    cx: float
    cy: float
    cz: float
    yaw: float
    hx: float = 0.0
    hy: float = 0.0
    r: float = 0.0

class WorldLoader:
    def __init__(self, world_file: str):
        self._world_file = world_file

    def _parse_pose(self, pose_str: str) -> Tuple[float, float, float, float, float, float]:
        vals = [float(v) for v in pose_str.split()]
        if len(vals) != 6:
            raise ValueError(f"Invalid pose string: {pose_str}")
        return tuple(vals)

    def _get_model_pose_xyyaw(self, model_elem: ET.Element) -> Tuple[float, float, float, float]:
        pose_el = model_elem.find("pose")
        if pose_el is None or pose_el.text is None:
            return 0.0, 0.0, 0.0, 0.0
        x, y, z, _r, _p, yaw = self._parse_pose(pose_el.text.strip())
        return x, y, z, yaw

    def _extract_footprint(self, model: ET.Element) -> Optional[Tuple[str, float, float]]:
        """
        Returns like:
        - ("box", hx, hy)
        - ("cylinder", radius, 0.0)
        """
        geom = model.find("./link/collision/geometry")
        if geom is None:
            return None
        children = list(geom)
        if not children:
            return None

        prim = children[0]
        if prim.tag == "box":
            size_el = prim.find("size")
            if size_el is None or size_el.text is None:
                return None
            sx, sy, _sz = [float(v) for v in size_el.text.strip().split()]
            return ("box", 0.5 * sx, 0.5 * sy)

        if prim.tag == "cylinder":
            radius_el = prim.find("radius")
            if radius_el is None or radius_el.text is None:
                return None
            return ("cylinder", float(radius_el.text.strip()), 0.0)

        return None

    def load_obstacles(
        self,
        obstacle_name_prefixes: Tuple[str, ...] = (
            "box_",
            "cylinder_",
            "wall_",
            "outer_wall_",
            "obstacle_",
            "macro_wall_",
            "maze_wall_",
        ),
    ) -> List[Obstacle]:
        tree = ET.parse(self._world_file)
        root = tree.getroot()
        world = root.find("world")
        if world is None:
            raise ValueError("No <world> element found in the SDF file.")

        obs: List[Obstacle] = []
        for model in world.findall("model"):
            name = model.get("name", "")
            if not _is_supported_obstacle_name(name, obstacle_name_prefixes):
                continue

            cx, cy, cz, yaw = self._get_model_pose_xyyaw(model)
            fp = self._extract_footprint(model)
            if fp is None:
                continue

            if fp[0] == "box":
                _, hx, hy = fp
                obs.append(Obstacle(name=name, kind="box", cx=cx, cy=cy, cz=cz, yaw=yaw, hx=hx, hy=hy))
            elif fp[0] == "cylinder":
                _, r, _ = fp
                obs.append(Obstacle(name=name, kind="cylinder", cx=cx, cy=cy, cz=cz, yaw=yaw, r=r))

        return obs

