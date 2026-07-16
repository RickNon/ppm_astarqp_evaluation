from dataclasses import dataclass
from typing import Optional


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class Box:
    size_x: float
    size_y: float


@dataclass
class Cylinder:
    radius: float


@dataclass
class Model:
    name: str
    pose: Pose
    geometry: object
    type: str
